"""Observe the real CPU teacher loop; no GPU, real data or backbone downloads."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from bridge_rgs import teacher

spec = importlib.util.spec_from_file_location(
    "capacity_stage", Path(__file__).parents[1] / "scripts/run_teacher_capacity_stage.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TinyBackbone(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.config = SimpleNamespace(model_type="dinov3_vit", patch_size=16,
                                      num_hidden_layers=4, hidden_size=width)
        self.proj = nn.Conv2d(3, width, 16, stride=16)

    def forward(self, pixel_values, output_hidden_states=True):
        patches = self.proj(pixel_values).flatten(2).transpose(1, 2)
        return SimpleNamespace(hidden_states=tuple(patches * (i+1) for i in range(5)))


@pytest.fixture
def fixture_data(tmp_path, monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    views = []
    for name, split, labeled in (("a", "train", True), ("b", "train", True),
                                 ("u", "train", False), ("v", "val", True)):
        image = np.zeros((64, 80, 3), np.uint8)
        image[:] = (60, 110, 150)
        image[20:40, 25:55] = (130, 90, 40)
        mask = np.zeros((64, 80), np.uint8)
        mask[:, 20:35] = 1
        mask[25:40, 40:55] = 2
        image_path, mask_path = tmp_path / f"{name}.png", tmp_path / f"{name}_mask.png"
        cv2.imwrite(str(image_path), image)
        cv2.imwrite(str(mask_path), mask)
        views.append({"name": f"{name}.png", "split": split, "image_path": str(image_path),
                      "mask_path": str(mask_path) if labeled else None, "valid_path": None,
                      "width": 80, "height": 64})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"class_names": ["background", "deck", "cable"], "views": views}))

    def loader(*args, pixel_profile="legacy_mixed_v1", **kwargs):
        model = teacher.DINOv3Teacher(TinyBackbone(), 3, 16)
        model.pixel_protocol = pixel_profile
        return model

    monkeypatch.setattr(teacher, "load_teacher", loader)
    config = teacher.TeacherConfig(steps=5, crop_size=64, channels=16, device="cpu", cpu_threads=1,
                                    consistency_start=2, context_start=1, context_probability=.5,
                                    context_short_side=64, eval_every=5, log_every=5, val_stride=48,
                                    checkpoint_every=1, independent_augmentation_rng=True, seed=73)
    yield manifest, model_dir, config
    torch.set_num_threads(old_threads)


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for x, y in zip(first, second, strict=True):
            assert_nested_equal(x, y)
    else:
        assert first == second


def test_trace_preserves_actual_cpu_trajectory_and_excludes_validation(fixture_data, tmp_path):
    manifest, model_dir, config = fixture_data
    teacher.train_teacher(manifest, model_dir, tmp_path / "plain", config)
    with runner.StageTrace(tmp_path / "steps.jsonl").installed() as trace:
        teacher.train_teacher(manifest, model_dir, tmp_path / "traced", config)
    plain = torch.load(tmp_path / "plain/last.pt", weights_only=False)
    traced = torch.load(tmp_path / "traced/last.pt", weights_only=False)
    for key in ("decoder", "ema_decoder", "optimizer", "numpy_generator_state", "torch_rng_state"):
        assert_nested_equal(plain[key], traced[key])
    records = runner.read_records(tmp_path / "steps.jsonl")
    assert [row["step"] for row in records] == [1, 2, 3, 4, 5]
    events = [event for row in records for event in row["events"]]
    assert not any(event.get("name") == "v.png" for event in events)
    assert {event["kind"] for event in events} == {"read", "crop", "context", "photometric"}
    assert sum(event["kind"] == "photometric" for event in events) == 5 + 2*4
    summary = trace.summary()
    assert summary["last_numpy_after"] == runner.checkpoint_rng(traced)["numpy"]
    assert summary["last_optimizer_rng_after"] == runner.checkpoint_rng(traced)["torch"]
    assert summary["pending_event_count"] == 0


def test_different_head_widths_keep_complete_sample_and_augmentation_trace(fixture_data, tmp_path, monkeypatch):
    manifest, model_dir, config = fixture_data
    traces, endpoints = [], []
    for width in (16, 32):
        def loader(*args, width=width, pixel_profile="legacy_mixed_v1", **kwargs):
            torch.rand(width * 13)  # Different architecture construction RNG consumption.
            model = teacher.DINOv3Teacher(TinyBackbone(width), 3, 16)
            model.pixel_protocol = pixel_profile
            return model

        monkeypatch.setattr(teacher, "load_teacher", loader)
        with runner.StageTrace(tmp_path / f"trace{width}.jsonl").installed() as trace:
            teacher.train_teacher(manifest, model_dir, tmp_path / f"model{width}", config)
        traces.append(trace.summary())
        endpoints.append(runner.checkpoint_rng(torch.load(tmp_path / f"model{width}/last.pt", weights_only=False)))
    assert traces[0]["stream_sha256"] == traces[1]["stream_sha256"]
    assert endpoints[0] == endpoints[1]


def test_wrappers_restore_on_error_and_reject_val_read_outside_evaluation(tmp_path):
    before = (teacher.ViewReader.read, teacher.aligned_crop, teacher.photometric_augment,
              teacher.evaluate_teacher, torch.optim.AdamW.step)
    with (pytest.raises(ValueError, match="Non-TRAIN"),
          runner.StageTrace(tmp_path / "trace.jsonl").installed()):
        teacher.ViewReader(3).read({"name": "forbidden", "split": "val"})
    assert before == (teacher.ViewReader.read, teacher.aligned_crop, teacher.photometric_augment,
                      teacher.evaluate_teacher, torch.optim.AdamW.step)


def test_recovery_keeps_original_tail_and_uses_checkpoint_prefix(tmp_path):
    identity = {"fixed": "contract"}
    for number, start, steps in ((0, 0, [1, 2, 3, 4]), (1, 2, [3])):
        directory = tmp_path / f"attempt_{number:03d}"
        directory.mkdir()
        (directory / "attempt.json").write_text(json.dumps({"identity": identity, "start_step": start}))
        (directory / "steps.jsonl").write_bytes(b"".join(
            runner.canonical({"step": step, "attempt": number}) + b"\n" for step in steps
        ) + (b'{"interrupted"' if number == 1 else b""))
    original = (tmp_path / "attempt_000/steps.jsonl").read_bytes()
    prefix = runner.recovered_prefix(tmp_path, 3, identity)
    assert [row["attempt"] for row in prefix] == [0, 0, 1]
    assert (tmp_path / "attempt_000/steps.jsonl").read_bytes() == original
    with pytest.raises(ValueError, match="ahead"):
        runner.recovered_prefix(tmp_path, 4, identity)
    with pytest.raises(ValueError, match="identity"):
        runner.recovered_prefix(tmp_path, 3, {"changed": "contract"})


def make_completed(audit_root, capacity, records, *, rng="same"):
    identity = {"capacity_config": capacity}
    directory = audit_root / "attempt_000"
    directory.mkdir(parents=True)
    (directory / "attempt.json").write_text(json.dumps({"identity": identity, "start_step": 0}))
    payload = b"".join(runner.canonical(row) + b"\n" for row in records)
    (directory / "steps.jsonl").write_bytes(payload)
    checkpoint = audit_root / "last.pt"
    checkpoint.write_bytes(b"synthetic endpoint")
    value = {"status": "completed", "capacity": capacity, "stage": "real", "steps": len(records),
             "plan_sha256": "plan", "runner_sha256": "runner", "source_sha256": {},
             "audit_root": str(audit_root), "trace_identity": identity,
             "trace": {"stream_sha256": hashlib.sha256(payload).hexdigest()},
             "last_checkpoint": str(checkpoint), "last_checkpoint_sha256": runner.file_sha(checkpoint),
             "terminal_rng": {"numpy": rng, "torch": {"cpu": rng, "cuda": []}}}
    path = audit_root / "stage_receipt.json"
    path.write_text(json.dumps(value))
    return path


def test_pair_comparison_catches_order_rng_and_trace_tampering(tmp_path):
    a = make_completed(tmp_path / "a", "hplus", [{"step": 1, "view": "a"}])
    b = make_completed(tmp_path / "b", "vit7b", [{"step": 1, "view": "a"}])
    assert runner.compare_stage_receipts(a, b)["status"] == "passed"
    c = make_completed(tmp_path / "c", "vit7b", [{"step": 1, "view": "b"}])
    comparison = runner.compare_stage_receipts(a, c)
    assert comparison["status"] == "failed" and comparison["first_mismatch_step"] == 1
    d = make_completed(tmp_path / "d", "vit7b", [{"step": 1, "view": "a"}], rng="different")
    assert runner.compare_stage_receipts(a, d)["status"] == "failed"
    (tmp_path / "b/attempt_000/steps.jsonl").write_text('{"step":1,"view":"tampered"}\n')
    with pytest.raises(ValueError, match="Persisted trace"):
        runner.compare_stage_receipts(a, b)


def test_checkpoint_change_after_completion_is_rejected(tmp_path):
    a = make_completed(tmp_path / "a", "hplus", [{"step": 1}])
    b = make_completed(tmp_path / "b", "vit7b", [{"step": 1}])
    (tmp_path / "b/last.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="endpoint"):
        runner.compare_stage_receipts(a, b)


@pytest.mark.parametrize("kind", ["G", "C", "C+G", ""])
def test_gpu_guard_allows_only_explicit_desktop_graphics(kind):
    xml = f"<nvidia_smi_log><gpu><processes><process_info><type>{kind}</type>" \
          "</process_info></processes></gpu></nvidia_smi_log>"
    if kind == "G":
        runner.verify_gpu_idle(xml)
    else:
        with pytest.raises(ValueError, match="compute or unknown"):
            runner.verify_gpu_idle(xml)


@pytest.mark.parametrize("xml", ["<nvidia_smi_log/>",
    "<nvidia_smi_log><gpu/></nvidia_smi_log>",
    "<nvidia_smi_log><gpu><processes>N/A</processes></gpu></nvidia_smi_log>",
    "<nvidia_smi_log><gpu><processes>Not Supported</processes></gpu></nvidia_smi_log>"])
def test_gpu_guard_fails_closed_for_unavailable_status(xml):
    with pytest.raises(ValueError):
        runner.verify_gpu_idle(xml)
