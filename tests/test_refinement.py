"""Checkpoint, correction-capacity and gradient contracts for refinement."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from bridge_rgs.model import GaussianScene, RefinementHead
from bridge_rgs.refinement import (
    MultiScaleRefinementHead,
    PanoramaContext,
    normalize_refiner_config,
)


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 2))
    yield
    torch.set_num_threads(previous)


def evidence(height=33, width=49, features=8, gradients=False):
    generator = torch.Generator().manual_seed(17)
    arrays = [torch.rand(height, width, channels, generator=generator)
              for channels in (features, 3, 1, 1)]
    probabilities = torch.randn(height, width, 5, generator=generator).softmax(-1)
    return [x.requires_grad_(gradients) for x in [*arrays, probabilities]]


def test_legacy_is_bitwise_equivalent_and_strictly_loadable():
    head = RefinementHead(feature_dim=8)
    with torch.no_grad():
        head.out.weight.normal_()
        head.out.bias.normal_()
    features, rgb, depth, alpha, probabilities = evidence()
    normalized_depth = torch.log1p(depth.detach().clamp_min(0))
    normalized_depth /= normalized_depth.amax().clamp_min(1)
    x = torch.cat([features, rgb.detach(), normalized_depth, alpha.detach()], -1).permute(2, 0, 1)[None]
    context = head.context(F.avg_pool2d(x, 4, ceil_mode=True))
    context = F.interpolate(context, x.shape[-2:], mode="bilinear", align_corners=False)
    original = (2 * head.out(torch.cat([head.local(x), context], 1))).tanh()[0].permute(1, 2, 0)
    actual = head(features, rgb, depth, alpha, p3d=probabilities)
    torch.testing.assert_close(actual, original, atol=0, rtol=0)
    restored = RefinementHead(feature_dim=8)
    restored.load_state_dict(head.state_dict(), strict=True)
    torch.testing.assert_close(restored(*evidence()), actual, atol=0, rtol=0)
    assert normalize_refiner_config() == {"type": "legacy"}
    assert normalize_refiner_config({"type": "legacy", "residual_bound": 1}) == {"type": "legacy"}


@pytest.mark.parametrize("context", ["none", "pyramid", "pyramid_strip"])
def test_new_head_starts_as_exact_zero_correction_on_odd_grid(context):
    head = MultiScaleRefinementHead(feature_dim=8, channels=16, context=context)
    values = evidence(height=35, width=53)
    residual = head(*values)
    assert residual.shape == values[-1].shape
    assert torch.count_nonzero(residual) == 0
    torch.testing.assert_close((values[-1].log() + residual).softmax(-1), values[-1])


def test_six_logit_bound_can_correct_confident_wrong_3d_prior():
    head = MultiScaleRefinementHead(feature_dim=8, channels=16, residual_bound=6.)
    features, rgb, depth, alpha, probabilities = evidence()
    probabilities[:] = torch.tensor([.98, .005, .005, .005, .005])
    with torch.no_grad():
        head.out.bias[2] = 12.
    residual = head(features, rgb, depth, alpha, probabilities)
    result = (probabilities.log() + residual).softmax(-1)
    assert torch.all(result.argmax(-1) == 2)
    assert residual.abs().max() <= 6
    # Even the optimal old +/-1 correction cannot overcome this prior gap.
    old_best = probabilities.log().clone()
    old_best[..., 0] -= 1
    old_best[..., 2] += 1
    assert torch.all(old_best.argmax(-1) == 0)


def test_auxiliary_render_channels_cannot_backpropagate_into_geometry():
    head = MultiScaleRefinementHead(feature_dim=8, channels=16)
    with torch.no_grad():
        head.out.weight.normal_(std=.1)
    features, rgb, depth, alpha, probabilities = evidence(gradients=True)
    residual = head(features, rgb, depth, alpha, probabilities)
    targets = torch.arange(residual.shape[1])[None].expand(residual.shape[0], -1) % 5
    loss = F.cross_entropy((probabilities.log() + residual).permute(2, 0, 1)[None], targets[None])
    loss.backward()
    assert features.grad is not None and features.grad.abs().sum() > 0
    assert probabilities.grad is not None and probabilities.grad.abs().sum() > 0
    assert rgb.grad is None and depth.grad is None and alpha.grad is None
    for parameter in head.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_panorama_global_branch_sees_whole_render():
    module = PanoramaContext(48, 16, mode="pyramid_strip")
    captured = []
    handle = module.pooled[0].register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0].detach()))
    x = torch.zeros(1, 48, 7, 11)
    x[..., -1, -1] = 77.
    result = module(x)
    handle.remove()
    torch.testing.assert_close(captured[0], torch.ones(1, 48, 1, 1))
    assert result.shape == (1, 16, 7, 11)
    assert torch.isfinite(result).all()


def test_new_scene_config_and_legacy_warmstart_preserve_all_3d_parameters():
    points = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 2]], dtype=np.float32)
    colors = np.full_like(points, .5)
    legacy = GaussianScene(points, colors, feature_dim=8, sh_degree=1)
    assert legacy.refiner_config == {"type": "legacy"}
    strong = GaussianScene(points, colors, feature_dim=8, sh_degree=1,
                           refiner_config={"type": "multiscale", "channels": 16})
    assert strong.refiner_config == {"type": "multiscale", "channels": 16,
                                      "residual_bound": 6., "context": "pyramid_strip"}
    retained = {key: value for key, value in legacy.state_dict().items() if not key.startswith("refiner.")}
    result = strong.load_state_dict(retained, strict=False)
    assert not result.unexpected_keys
    assert result.missing_keys and all(key.startswith("refiner.") for key in result.missing_keys)
    for key, value in retained.items():
        torch.testing.assert_close(strong.state_dict()[key], value, atol=0, rtol=0)
    reloaded = GaussianScene(points, colors, feature_dim=8, sh_degree=1,
                             refiner_config=strong.refiner_config)
    reloaded.load_state_dict(strong.state_dict(), strict=True)


def test_load_scene_restores_new_architecture_and_legacy_without_metadata(tmp_path):
    from bridge_rgs.train import load_scene
    points = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 2]], dtype=np.float32)
    colors = np.full_like(points, .5)
    for config in (None, {"type": "multiscale", "channels": 16, "context": "pyramid"}):
        scene = GaussianScene(points, colors, feature_dim=8, sh_degree=1, refiner_config=config)
        values = {"model": scene.state_dict(), "feature_dim": 8, "sh_degree": 1,
                  "scene_scale": scene.scene_scale}
        if config is not None:
            values["refiner_config"] = scene.refiner_config
        path = tmp_path / ("legacy.pt" if config is None else "multiscale.pt")
        torch.save(values, path)
        restored, _state = load_scene(path, device="cpu")
        assert restored.refiner_config == scene.refiner_config
        for name, value in scene.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)


@pytest.mark.parametrize("config", [
    {"type": "multiscale", "context": "typo"}, {"type": "unknown"},
    {"type": "legacy", "channels": 64}, {"type": "multiscale", "channels": 7},
    {"type": "multiscale", "residual_bound": float("nan")},
])
def test_architecture_configuration_does_not_silently_ignore_errors(config):
    with pytest.raises(ValueError):
        normalize_refiner_config(config)
