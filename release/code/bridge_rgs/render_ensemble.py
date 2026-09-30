"""The measured fixed DINOv3/scene combination, from rendered RGB only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .coordinates import protocol_metadata, require_matching_protocol
from .data import CLASS_NAMES
from .teacher import (
    checkpoint_adapter_options,
    checkpoint_pixel_protocol,
    file_sha256,
    load_checkpoint_adapters,
    load_teacher,
    predict_image,
)

FIXED_INFERENCE = {"tile_size": 768, "stride": 512, "flip": True,
                   "context_weight": .25, "context_short_side": 768}


class FixedRenderedTeacher:
    """Load once; use the predeclared 50/50 probability mixture for each camera.

    Teacher training and inference renderer identities are recorded separately.
    A different renderer is allowed, as in the explicit renderer-transfer study;
    a different pixel convention or known training split is rejected.
    """

    def __init__(self, teacher_checkpoint, scene_checkpoint, scene_state, device="cuda"):
        checkpoint = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
        source = checkpoint["provenance"]
        profile = checkpoint_pixel_protocol(checkpoint)
        require_matching_protocol(profile, scene_state, "teacher/scene rendering")
        if source["class_names"] != CLASS_NAMES:
            raise ValueError("Teacher class IDs differ from the bridge rendering schema")
        domain = source.get("image_source_protocol") or {}
        teacher_manifest_hash = domain.get("original_manifest_sha256", source["manifest_sha256"])
        scene_manifest_hash = scene_state.get("manifest_sha256")
        if scene_manifest_hash is None:
            historical_manifest = scene_state.get("config", {}).get("manifest")
            if historical_manifest and Path(historical_manifest).is_file():
                scene_manifest_hash = file_sha256(historical_manifest)
        if scene_manifest_hash and teacher_manifest_hash != scene_manifest_hash:
            raise ValueError("Teacher and scene training manifest SHA differ")
        model_dir = Path(source["model_dir"])
        if file_sha256(model_dir / "config.json") != source["model_config_sha256"]:
            raise ValueError("Teacher backbone configuration changed")
        weights = {path.name: file_sha256(path) for path in sorted(model_dir.glob("*.safetensors"))}
        if not weights or weights != source["model_weights_sha256"]:
            raise ValueError("Teacher backbone weights changed")
        indexes = {path.name: file_sha256(path) for path in sorted(model_dir.glob("*.safetensors.index.json"))}
        if indexes != source.get("model_index_sha256", {}):
            raise ValueError("Teacher sharded backbone index is missing from provenance or changed")
        self.input_files = {
            str(Path(teacher_checkpoint).resolve()): file_sha256(teacher_checkpoint),
            str((model_dir / "config.json").resolve()): source["model_config_sha256"],
            **{str((model_dir / name).resolve()): value for name, value in (weights | indexes).items()},
        }
        self.model = load_teacher(model_dir, len(CLASS_NAMES), checkpoint["configuration"]["channels"],
                                  device=str(device), pixel_profile=profile,
                                  **checkpoint_adapter_options(checkpoint["configuration"]))
        self.model.decoder.load_state_dict(checkpoint["ema_decoder"])
        load_checkpoint_adapters(self.model, checkpoint)
        self.model.eval().requires_grad_(False)
        renderer_sha = file_sha256(scene_checkpoint)
        self.receipt = {
            "protocol": "fixed_rendered_teacher_0.5_v1", "teacher_weight": .5,
            "inference": dict(FIXED_INFERENCE), "pixel_protocol": protocol_metadata(profile),
            "teacher_checkpoint": str(Path(teacher_checkpoint).resolve()),
            "teacher_checkpoint_sha256": file_sha256(teacher_checkpoint),
            "teacher_modelscope_backbone": {"model_dir": str(model_dir.resolve()),
                                            "config_sha256": source["model_config_sha256"],
                                            "weights_sha256": weights, "index_sha256": indexes},
            "input_files_sha256": self.input_files,
            "teacher_training_renderer_sha256": domain.get("renderer_checkpoint_sha256"),
            "inference_renderer_sha256": renderer_sha,
            "teacher_training_manifest_sha256": teacher_manifest_hash,
            "scene_training_manifest_sha256": scene_manifest_hash,
            "input": "uint8 RGB from this scene's pinhole render; no external image or label",
            "fusion_grid": "pinhole canvas, before official distortion remapping",
            "scope": "Fixed costly engineering combination. Native development metrics do not score official overscan exports.",
        }

    def verify_inputs(self):
        """Detect dependency changes over a completed prediction/evaluation run."""
        for path, expected in self.input_files.items():
            if file_sha256(path) != expected:
                raise ValueError(f"Teacher input changed during inference: {path}")

    @torch.inference_mode()
    def blend(self, rgb, probabilities):
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or probabilities.shape != (*rgb.shape[:2], 5):
            raise ValueError("Teacher mixture requires aligned HWC RGB and five-class probabilities")
        image = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
        prediction, _ = predict_image(self.model, image, **FIXED_INFERENCE)
        if prediction.shape != (5, *rgb.shape[:2]) or not np.isfinite(prediction).all():
            raise ValueError("Teacher returned invalid probabilities")
        # Same quantization and fixed arithmetic as the renderer-transfer study.
        return .5 * probabilities + .5 * prediction.transpose(1, 2, 0)
