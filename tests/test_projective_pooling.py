"""CPU-only contracts for the isolated, parameter-preserving context adapter."""
import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from bridge_rgs.projective_pooling import (
    AXIS,
    OFFSETS,
    ROLL_90,
    attach_projective_context,
    prepare_sampling_geometry,
    sampled_average,
)
from bridge_rgs.refinement import PanoramaContext
from bridge_rgs.structure_axes import feature_centers, project_direction, reflect_intrinsics


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def camera(width=257, height=193):
    return torch.tensor([[230., 0., width/2], [0., 240., height/2], [0., 0., 1.]]), torch.eye(4)


class TinyScene(nn.Module):
    def __init__(self, width=257, height=193):
        super().__init__()
        self.refiner = nn.Module()
        self.refiner.panorama = PanoramaContext(16, 16)
        self.features = nn.Parameter(torch.randn(1, 16, (height+15)//16, (width+15)//16))
        self.rgb = nn.Parameter(torch.randn(3))
        self.calls = []

    def render(self, K, w2c, width, height, *, degree=None, semantics=True):
        self.calls.append((K, w2c, width, height, degree, semantics))
        return {"rgb": self.rgb * 2, "raw": self.features.mean(1),
                "final": self.refiner.panorama(self.features)}


def horizontal_rotation():
    a = AXIS / np.linalg.norm(AXIS)
    b = np.cross([0., 1., 0.], a)
    b /= np.linalg.norm(b)
    return np.stack([a, b, np.cross(a, b)])


@pytest.mark.parametrize("mode", ["original", "per_camera", "projective", "wrong"])
def test_step_zero_exact_forward_gradients_and_parameter_state(mode):
    torch.manual_seed(7)
    scene = TinyScene()
    baseline = copy.deepcopy(scene)
    before_keys = tuple(scene.state_dict())
    before_ids = [id(p) for p in scene.parameters()]
    before = {k: v.clone() for k, v in scene.state_dict().items()}
    adapter = attach_projective_context(scene, {"mode": mode})
    K, pose = camera()
    actual, expected = scene.render(K, pose, 257, 193), baseline.render(K, pose, 257, 193)
    assert all(torch.equal(actual[k], expected[k]) for k in actual)
    actual["final"].square().mean().backward()
    expected["final"].square().mean().backward()
    for p, q in zip(scene.parameters(), baseline.parameters()):
        assert (p.grad is None and q.grad is None) or torch.equal(p.grad, q.grad)
    assert tuple(scene.state_dict()) == before_keys
    assert [id(p) for p in scene.parameters()] == before_ids
    assert all(torch.equal(v, before[k]) for k, v in scene.state_dict().items())
    json.dumps(adapter.configuration, allow_nan=False)
    adapter.detach()
    adapter.detach()
    assert "render" not in scene.__dict__
    assert "forward" not in scene.refiner.panorama.__dict__
    assert not hasattr(scene, "_projective_context_adapter")
    assert torch.equal(scene.render(K, pose, 257, 193)["final"], expected["final"])


@pytest.mark.parametrize("mode", ["per_camera", "projective", "wrong"])
@pytest.mark.parametrize("step", [250, 500, 1999])
def test_active_forward_finite_gradient_and_render_camera_unmodified(mode, step):
    torch.manual_seed(11)
    scene = TinyScene()
    adapter = attach_projective_context(scene, {"mode": mode})
    adapter.set_step(step)
    K, pose = camera()
    K.requires_grad_()
    pose.requires_grad_()
    saved = (K.detach().clone(), pose.detach().clone())
    out = scene.render(K, pose, width=257, height=193, degree=3, semantics=True)
    out["final"].square().mean().backward()
    assert scene.calls[-1][0] is K and scene.calls[-1][1] is pose
    assert scene.calls[-1][2:] == (257, 193, 3, True)
    assert torch.equal(K, saved[0]) and torch.equal(pose, saved[1])
    assert K.grad is None and pose.grad is None
    assert torch.isfinite(out["final"]).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in scene.refiner.parameters())
    assert torch.isfinite(scene.features.grad).all()
    assert adapter._camera is None
    assert adapter.diagnostics["blend"] == min(step/500, 1)
    assert adapter.diagnostics["sampling_applied"]
    json.dumps(adapter.diagnostics, allow_nan=False)


def test_only_horizontal_branch_changed_and_mix_is_after_projection():
    torch.manual_seed(23)
    scene = TinyScene()
    adapter = attach_projective_context(scene, {"mode": "projective"})
    seen = []
    hook = scene.refiner.panorama.merge.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone()))
    K, pose = camera()
    for step in (0, 250, 500):
        adapter.set_step(step)
        scene.render(K, pose, 257, 193)
    hook.remove()
    old, mixed, new = seen
    start, end = 16+4*8, 16+5*8
    for a in (mixed, new):
        assert torch.equal(a[:, :start], old[:, :start])
        assert torch.equal(a[:, end:], old[:, end:])
    assert torch.equal(mixed[:, start:end], .5*old[:, start:end]+.5*new[:, start:end])
    assert not torch.equal(new[:, start:end], old[:, start:end])


def test_common_support_is_per_offset_intersection_and_coordinates():
    K, _ = camera(513, 513)
    R = horizontal_rotation()
    geometry = prepare_sampling_geometry(K, R, 513, 513)
    masks = geometry["own_valid"]
    assert np.array_equal(geometry["common_valid"],
                          masks["per_camera"] & masks["projective"] & masks["wrong"])
    assert np.any(masks["projective"] != geometry["common_valid"])
    assert geometry["grids"]["projective"].dtype == np.float32
    centers = feature_centers(513, 513)
    d, _, _ = project_direction(K.numpy(), R, AXIS, centers)
    dw, _, _ = project_direction(K.numpy(), ROLL_90 @ R, AXIS, centers)
    for mode, directions in (("projective", d), ("wrong", dw)):
        grid = geometry["grids"][mode].astype(np.float64)
        recovered = (grid+1)/2*32*16+.5
        expected = centers[..., None, :] + OFFSETS[:, None]*directions[..., None, :]
        np.testing.assert_allclose(recovered, expected, atol=4e-5, rtol=0)
    np.testing.assert_allclose(abs(d[..., 0]), 1., atol=1e-12)
    np.testing.assert_allclose(abs(dw[..., 1]), 1., atol=1e-12)
    diag = geometry["diagnostics"]
    assert diag["total_queries"] == 33*33
    assert diag["nonempty_queries"] == 33*33
    assert diag["common_valid_samples"] == int(geometry["common_valid"].sum())
    assert diag["total_offset_slots"] == 33*33*17
    assert diag["true_wrong_comparable_queries"] == 33*33
    assert diag["true_wrong_unoriented_degrees_median"] == pytest.approx(90.)
    assert diag["true_wrong_unoriented_degrees_p95"] == pytest.approx(90.)


def test_bilinear_sampling_matches_independent_linear_ramp_with_common_counts():
    K, pose = camera(257, 193)
    geometry = prepare_sampling_geometry(K, pose[:3, :3], 257, 193)
    h, w = geometry["diagnostics"]["feature_shape"]
    yy, xx = np.mgrid[:h, :w]
    x = torch.tensor((xx+10*yy)[None, None], dtype=torch.float64, requires_grad=True)
    mask = geometry["common_valid"]
    for mode, grid in geometry["grids"].items():
        result, empty = sampled_average(x, torch.from_numpy(grid), torch.from_numpy(mask))
        xy = (grid.astype(np.float64)+1)/2*np.array([w-1, h-1])
        values = xy[..., 0]+10*xy[..., 1]
        counts = mask.sum(-1)
        expected = (values*mask).sum(-1)/np.maximum(counts, 1)
        expected[counts == 0] = np.broadcast_to(x.detach().numpy()[0, 0].mean(-1)[:, None],
                                               (h, w))[counts == 0]
        np.testing.assert_allclose(result.detach().numpy()[0, 0], expected, atol=1e-12)
        assert torch.equal(empty, torch.from_numpy(counts == 0))
        assert mode in {"projective", "wrong", "per_camera"}


def test_degenerate_all_empty_and_singleton_fallback_exact():
    torch.manual_seed(5)
    scene = TinyScene(1, 1)
    K = torch.tensor([[1., 0., .5], [0., 1., .5], [0., 0., 1.]])
    # Map fixed axis to optical z: sole ray is exactly parallel to the axis.
    R = horizontal_rotation()[[1, 2, 0]]
    pose = torch.eye(4)
    pose[:3, :3] = torch.from_numpy(R)
    expected = scene.render(K, pose, 1, 1)
    adapter = attach_projective_context(scene, {"mode": "projective"})
    adapter.set_step(500)
    actual = scene.render(K, pose, 1, 1)
    assert torch.equal(actual["final"], expected["final"])
    assert adapter.diagnostics["empty_fraction"] == 1
    assert adapter.diagnostics["per_camera_constant"] is None
    assert all(np.isfinite(g).all() for g in adapter._geometry["grids"].values())


def test_novel_pose_changes_field_without_name_and_restores_nested_camera():
    scene = TinyScene()
    adapter = attach_projective_context(scene, {"mode": "projective"})
    adapter.set_step(500)
    K, pose = camera()
    adapter.set_camera(K, pose, 257, 193)
    previous = adapter._camera
    first = scene.render(K, pose, 257, 193)["final"]
    new_pose = pose.clone()
    new_pose[:3, :3] = torch.from_numpy(ROLL_90)
    second = scene.render(K, new_pose, 257, 193)["final"]
    assert adapter._camera is previous
    assert not torch.equal(first, second)
    # Translation changes the renderer's camera but not a world direction's VP.
    translated = new_pose.clone()
    translated[:3, 3] = torch.tensor([3., 2., 1.])
    assert torch.equal(scene.render(K, translated, 257, 193)["final"], second)


def test_mirrored_native_grid_is_recomputed_not_coarse_flipped():
    K, pose = camera(1320, 989)
    original = prepare_sampling_geometry(K, pose[:3, :3], 1320, 989)
    mirrored = prepare_sampling_geometry(reflect_intrinsics(K.numpy(), 1320),
                                          pose[:3, :3], 1320, 989)
    assert original["diagnostics"]["feature_shape"] == [62, 83]
    center = len(OFFSETS)//2
    grid = mirrored["grids"]["projective"]
    native = (grid[..., center, 0].astype(np.float64)+1)/2*82*16+.5
    np.testing.assert_allclose(native[0], .5+16*np.arange(83), atol=4e-5)
    # Reflection of old last coarse center is 7.5, new first center is .5.
    assert 1320-(.5+16*82)-native[0, 0] == 7


def test_empty_query_has_exact_old_full_branch_not_zero_placeholder():
    scene = TinyScene()
    adapter = attach_projective_context(scene, {"mode": "projective"})
    adapter.set_step(500)
    K, pose = camera()
    adapter.set_camera(K, pose, 257, 193)
    grid, support = adapter._sampling_tensors(scene.features)
    # Synthetic mask drives one empty query while preserving the rest.
    support = support.clone()
    support[3, 4] = False
    adapter._device_cache = (adapter._device_cache[0], grid, support)
    captures = []
    hook = scene.refiner.panorama.merge.register_forward_pre_hook(
        lambda _m, args: captures.append(args[0].detach().clone()))
    adapter.set_step(0)
    scene.refiner.panorama(scene.features)
    adapter.set_step(500)
    scene.refiner.panorama(scene.features)
    hook.remove()
    assert torch.equal(captures[0][:, 48:56, 3, 4], captures[1][:, 48:56, 3, 4])


def test_fail_closed_on_bad_configuration_camera_or_feature_shape():
    scene = TinyScene()
    with pytest.raises(ValueError, match="Expected mode"):
        attach_projective_context(scene, {"mode": "new"})
    with pytest.raises(ValueError, match="Expected mode"):
        attach_projective_context(scene, {"mode": "original", "offsets": [0]})
    adapter = attach_projective_context(scene, {"mode": "wrong"})
    with pytest.raises(ValueError, match="already"):
        attach_projective_context(scene, {"mode": "wrong"})
    with pytest.raises(ValueError, match="step"):
        adapter.set_step(True)
    adapter.set_step(500)
    with pytest.raises(RuntimeError, match="actual camera"):
        scene.refiner.panorama(scene.features)
    K, pose = camera()
    with pytest.raises(ValueError, match="homogeneous"):
        adapter.set_camera(K, torch.zeros(4, 4), 257, 193)
    with pytest.raises(ValueError, match="stride16"):
        scene.render(K, pose, 273, 193)
    assert adapter._camera is None
    with pytest.raises(ValueError, match="width"):
        prepare_sampling_geometry(K, pose[:3, :3], 0, 193)


def test_sampled_average_rejects_gradient_geometry_and_wrong_support():
    x = torch.ones(1, 2, 3, 4)
    grid = torch.zeros(3, 4, 17, 2, requires_grad=True)
    mask = torch.ones(3, 4, 17, dtype=torch.bool)
    with pytest.raises(ValueError, match="detached"):
        sampled_average(x, grid, mask)
    with pytest.raises(ValueError, match="lattice"):
        sampled_average(x, grid.detach(), mask.float())
