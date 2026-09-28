"""Density diagnostics and shuffled camera coverage survive checkpointing."""

import io

import cv2
import numpy as np
import pytest
import torch

from bridge_rgs.train import (
    density_checkpoint_state,
    initialize_density_extras,
    initialize_view_sampler,
    rendering_sh_degree,
    restore_density_extras,
    sample_training_view,
    scheduled_gaussian_budget,
    training_semantic_weights,
)


def test_hybrid_window_and_sampler_roundtrip():
    extras = initialize_density_extras(5, 3, "cpu")
    extras["residual_scores"][:] = torch.arange(5)
    extras["camera_quality"][:] = torch.tensor([1., .4, .8])
    extras["residual_observed_views"][2] = torch.tensor([True, False, True, False, True])
    extras.update(sampler_order=[2, 0, 1], sampler_cursor=2, pause_until=400)
    state = density_checkpoint_state(torch.ones(5), torch.ones(5), torch.ones(5), {}, None, extras)
    stream = io.BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    restored = restore_density_extras(torch.load(stream, weights_only=False)["extras"], 5, 3, "cpu")
    torch.testing.assert_close(restored["residual_scores"], extras["residual_scores"])
    torch.testing.assert_close(restored["camera_quality"], extras["camera_quality"])
    assert restored["residual_observed_views"][2].tolist() == [True, False, True, False, True]
    assert restored["sampler_order"][restored["sampler_cursor"]] == 1
    assert restored["pause_until"] == 400


def test_legacy_density_checkpoint_initializes_new_diagnostics_conservatively():
    state = restore_density_extras(None, 5, 3, "cpu")
    assert state["residual_counts"].sum() == 0
    assert state["camera_quality"].tolist() == [1., 1., 1.]


def test_density_resume_rejects_corrupt_shapes_and_duplicate_camera_order():
    extras = initialize_density_extras(5, 3, "cpu")
    with pytest.raises(ValueError, match="Gaussian count"):
        restore_density_extras(extras, 6, 3, "cpu")
    extras["sampler_order"] = [0, 0, 1]
    with pytest.raises(ValueError, match="sampler"):
        restore_density_extras(extras, 5, 3, "cpu")


def test_frozen_rgb_warmstart_uses_full_sh_from_first_step():
    assert rendering_sh_degree(1, 3, {"freeze_rgb": True, "sh_interval": 1000}) == 3
    assert rendering_sh_degree(1, 3, {"sh_interval": 1000}) == 0
    assert rendering_sh_degree(5000, 3, {"sh_interval": 1000}) == 3


def test_capacity_schedule_reserves_growth_for_later_high_resolution_views():
    config = {"max_gaussians": 500000, "initial_gaussians": 60000,
              "densify_start": 1000, "densify_stop": 23000, "density_budget_schedule": "linear"}
    assert scheduled_gaussian_budget(1000, config) == 60000
    assert scheduled_gaussian_budget(12000, config) == 280000
    assert scheduled_gaussian_budget(30000, config) == 500000


@pytest.mark.parametrize("mode", ["shuffle", "random"])
def test_independent_view_sampler_ignores_teacher_rng_and_resumes_exactly(mode):
    config = {"independent_view_rng": True, "seed": 71}
    population = [0, 2, 4, 6]
    first = initialize_density_extras(5, 7, "cpu")
    rng_first = initialize_view_sampler(config, first)
    uninterrupted = [sample_training_view(population, mode, first, rng_first) for _ in range(20)]
    second = initialize_density_extras(5, 7, "cpu")
    rng_second = initialize_view_sampler(config, second)
    sequence = []
    for _ in range(7):
        np.random.choice(400, 71, replace=False)  # Stand-in for optional teacher/fusion sampling.
        sequence.append(sample_training_view(population, mode, second, rng_second))
    state = density_checkpoint_state(torch.ones(5), torch.ones(5), torch.ones(5), {}, None, second)
    stream = io.BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    restored = restore_density_extras(torch.load(stream, weights_only=False)["extras"], 5, 7, "cpu")
    rng_restored = initialize_view_sampler(config, restored)
    for _ in range(13):
        np.random.rand(23)
        sequence.append(sample_training_view(population, mode, restored, rng_restored))
    assert sequence == uninterrupted


def test_legacy_view_sampler_preserves_global_numpy_random_draws():
    np.random.seed(39)
    expected = [int(np.random.randint(6)) for _ in range(10)]
    np.random.seed(39)
    extras = initialize_density_extras(5, 6, "cpu")
    generator = initialize_view_sampler({}, extras)
    assert generator is None
    assert [sample_training_view(list(range(6)), "random", extras, generator) for _ in range(10)] == expected


def test_raw_semantic_weight_power_can_differ_and_defaults_to_final(tmp_path):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:1] = 2
    path = str(tmp_path / "mask.png")
    assert cv2.imwrite(path, mask)
    views = [{"mask_path": path}]
    final, raw = training_semantic_weights(views, {"class_weight_power": .25}, "cpu")
    assert final is raw
    final, raw = training_semantic_weights(views, {"class_weight_power": .25, "raw_class_weight_power": .5}, "cpu")
    assert raw[2] / raw[0] > final[2] / final[0]
    explicit = [1, 1, 2, 1, 1]
    final, raw = training_semantic_weights(views, {"class_weights": explicit, "raw_class_weight_power": .5}, "cpu")
    torch.testing.assert_close(final, raw)
