"""Frozen DINOv3 semantic teacher with multilevel fusion and aligned EMA consistency.

The teacher only trains on manifest views marked ``train``. Exported probabilities
use the original manifest image grid, and are never generated for validation views.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import OrderedDict
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .coordinates import (
    CORNER,
    LEGACY,
    annotate_manifest,
    pixel_protocol,
    protocol_metadata,
    require_matching_protocol,
)

IGNORE_LABEL = 255


def file_sha256(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def backbone_index_hashes(model_dir: str | Path) -> dict[str, str]:
    """Bind HF shard routing in addition to the actual tensor files."""
    return {path.name: file_sha256(path)
            for path in sorted(Path(model_dir).glob("*.safetensors.index.json"))}


def require_backbone_index(source: dict, actual: dict[str, str], context: str) -> None:
    # Historical unsharded H+ checkpoints have no index and remain compatible.
    if source.get("model_index_sha256", {}) != actual:
        raise ValueError(f"{context}: backbone index missing from provenance or changed")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _groups(channels: int) -> int:
    return math.gcd(channels, min(16, max(1, channels // 4)))


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(ConvBlock(channels, channels), ConvBlock(channels, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class SemanticDecoder(nn.Module):
    """DPT-style four-level fusion plus an image-aligned stride-four detail path."""

    def __init__(self, feature_dim: int, num_classes: int, channels: int = 192):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(feature_dim, channels, 1) for _ in range(4)])
        self.reassemble = nn.ModuleList([ConvBlock(channels, channels) for _ in range(4)])
        self.fusion = nn.ModuleList([ResidualBlock(channels) for _ in range(4)])
        self.detail = nn.Sequential(
            ConvBlock(3, channels // 2, 2),
            ConvBlock(channels // 2, channels, 2),
            ResidualBlock(channels),
        )
        self.detail_gate = nn.Conv2d(2 * channels, channels, 1)
        self.head = nn.Sequential(
            ConvBlock(channels, channels), nn.Conv2d(channels, num_classes, 1)
        )
        self.boundary_head = nn.Sequential(
            ConvBlock(channels, channels // 2), nn.Conv2d(channels // 2, 1, 1)
        )

    def forward(self, features: list[torch.Tensor], image: torch.Tensor) -> dict[str, torch.Tensor]:
        h, w = image.shape[-2:]
        maps = []
        for index, feature in enumerate(features):
            value = self.projections[index](feature.float())
            size = (max(1, h // (4 * 2**index)), max(1, w // (4 * 2**index)))
            maps.append(
                self.reassemble[index](
                    F.interpolate(value, size=size, mode="bilinear", align_corners=False)
                )
            )
        value = self.fusion[3](maps[3])
        for index in (2, 1, 0):
            value = self.fusion[index](
                maps[index]
                + F.interpolate(
                    value, size=maps[index].shape[-2:], mode="bilinear", align_corners=False
                )
            )
        detail = self.detail(image)
        detail = F.interpolate(detail, size=value.shape[-2:], mode="bilinear", align_corners=False)
        gate = torch.sigmoid(self.detail_gate(torch.cat((value, detail), dim=1)))
        value = value + gate * detail
        return {
            "logits": F.interpolate(
                self.head(value), size=(h, w), mode="bilinear", align_corners=False
            ),
            "boundary_logits": F.interpolate(
                self.boundary_head(value), size=(h, w), mode="bilinear", align_corners=False
            ),
        }


class DINOv3Teacher(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, channels: int = 192):
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)
        self.train_backbone_adapters = False
        self.pixel_protocol = LEGACY
        config = backbone.config
        if getattr(config, "model_type", "dinov3_vit") != "dinov3_vit":
            raise ValueError("A DINOv3 ViT checkpoint is required")
        self.patch_size = int(config.patch_size)
        self.feature_layers = [
            round(config.num_hidden_layers * ratio / 4) for ratio in (1, 2, 3, 4)
        ]
        self.decoder = SemanticDecoder(config.hidden_size, num_classes, channels)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def extract_features(self, image: torch.Tensor) -> list[torch.Tensor]:
        if image.shape[-2] % self.patch_size or image.shape[-1] % self.patch_size:
            raise ValueError("Teacher input dimensions must be divisible by its patch size")
        dtype = next(self.backbone.parameters()).dtype
        # A trainable adapter in the final blocks starts autograd only there:
        # frozen prefix parameters and input images never require gradients.
        context = nullcontext() if self.train_backbone_adapters else torch.no_grad()
        with context:
            hidden = self.backbone(
                pixel_values=((image - self.image_mean) / self.image_std).to(dtype),
                output_hidden_states=True,
            ).hidden_states
            patch_h, patch_w = (
                image.shape[-2] // self.patch_size,
                image.shape[-1] // self.patch_size,
            )
            count = patch_h * patch_w
            # Taking the final N tokens excludes CLS and all register tokens.
            return [
                F.layer_norm(hidden[index][:, -count:, :], (hidden[index].shape[-1],))
                .transpose(1, 2)
                .reshape(image.shape[0], -1, patch_h, patch_w)
                .contiguous()
                for index in self.feature_layers
            ]

    def forward(
        self, image: torch.Tensor, decoder: nn.Module | None = None
    ) -> dict[str, torch.Tensor]:
        return (self.decoder if decoder is None else decoder)(self.extract_features(image), image)


def load_teacher(
    model_dir: str | Path, num_classes: int, channels: int = 192, device: str = "cuda",
    *, adapter_rank: int = 0, adapter_blocks: int = 4, adapter_alpha: float = 16.0,
    pixel_profile: str = LEGACY,
) -> DINOv3Teacher:
    """Construct a backbone/head for an explicit grid; missing profile stays legacy."""
    profile = pixel_protocol(pixel_profile)
    from transformers import AutoModel

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    backbone = AutoModel.from_pretrained(
        str(model_dir), local_files_only=True, dtype=dtype, attn_implementation="sdpa"
    )
    model = DINOv3Teacher(backbone, num_classes, channels).to(device)
    model.pixel_protocol = profile
    if adapter_rank:
        from .teacher_adapters import install_dinov3_lora

        install_dinov3_lora(backbone, last_n=adapter_blocks, rank=adapter_rank, alpha=adapter_alpha)
        model.train_backbone_adapters = True
        model.to(device)
    return model


def checkpoint_adapter_options(configuration: dict) -> dict:
    """Old checkpoints have no adapters; their load path remains unchanged."""
    return {key: configuration.get(key, default) for key, default in (
        ("adapter_rank", 0), ("adapter_blocks", 4), ("adapter_alpha", 16.0)
    )}


def checkpoint_pixel_protocol(checkpoint: dict) -> str:
    """Resolve only declarations stored in this teacher checkpoint/provenance.

    Historical missing declarations mean legacy, regardless of the manifest file
    currently at the recorded path. Never infer a migration from today's data.
    New checkpoints duplicate the full declaration at root and in provenance;
    conflicting copies fail rather than preferring one silently.
    """
    records = [checkpoint]
    if "provenance" in checkpoint:
        records.append(checkpoint["provenance"])
    declarations = [pixel_protocol(record) for record in records if "pixel_protocol" in record]
    if not declarations:
        return LEGACY
    if len(set(declarations)) != 1:
        raise ValueError("Teacher checkpoint pixel protocol declarations disagree")
    return declarations[0]


def require_teacher_manifest_protocol(checkpoint: dict, manifest: dict, context: str) -> str:
    """Check checkpoint and every manifest view without mutating either input."""
    annotate_manifest(manifest)  # Also rejects conflicting per-view declarations.
    profile = checkpoint_pixel_protocol(checkpoint)
    require_matching_protocol(profile, manifest, context)
    return profile


def verify_teacher_render_protocol(manifest: dict, image_source_protocol: dict | None) -> None:
    """Check grid lineage in addition to validate_image_sources' byte/split audit.

    Original/derived manifests, both immutable render receipts, and the actual
    renderer checkpoint must share one profile. Missing old fields mean legacy.
    This is CPU metadata loading; no photo/annotation pixels are opened here.
    """
    annotate_manifest(manifest)
    if image_source_protocol is None:
        return
    from .teacher_domains import MULTI_COMPONENT_DOMAIN, verify_multi_component_renderers

    if image_source_protocol.get("id") == MULTI_COMPONENT_DOMAIN:
        # Explicit delivered-original-RGB -> legacy canvas migration. The raw
        # renderer profiles are verified separately, never relabeled legacy.
        verify_multi_component_renderers(image_source_protocol)
        return
    source_manifest = json.loads(Path(image_source_protocol["original_manifest"]).read_text())
    annotate_manifest(source_manifest)
    require_matching_protocol(source_manifest, manifest, "teacher render source manifest")
    require_matching_protocol(image_source_protocol, manifest, "teacher rendered image source")
    for split in ("train", "val"):
        receipt = json.loads(Path(image_source_protocol[f"{split}_render_receipt"]["path"]).read_text())
        require_matching_protocol(receipt, manifest, f"teacher {split} render receipt")
    renderer = torch.load(image_source_protocol["renderer_checkpoint"], map_location="cpu", weights_only=False)
    require_teacher_renderer_protocol(renderer, manifest, image_source_protocol["original_manifest_sha256"])


def require_teacher_renderer_protocol(renderer: dict, manifest: dict, manifest_sha256: str) -> str:
    """Reject a different renderer grid/data lineage before teacher RGB export."""
    annotate_manifest(manifest)
    require_matching_protocol(renderer, manifest, "teacher renderer checkpoint")
    profile = pixel_protocol(renderer)
    if (renderer.get("manifest_sha256") not in (None, manifest_sha256)
            or (profile == CORNER and renderer.get("manifest_sha256") != manifest_sha256)):
        raise ValueError("Teacher renderer checkpoint manifest SHA differs")
    return profile


def load_checkpoint_adapters(model: DINOv3Teacher, checkpoint: dict, *, ema: bool = True) -> None:
    # This check applies to the head-only load path as well as LoRA checkpoints.
    require_matching_protocol(getattr(model, "pixel_protocol", None), checkpoint_pixel_protocol(checkpoint),
                              "teacher weight loading")
    if model.train_backbone_adapters:
        from .teacher_adapters import load_adapter_state_dict

        load_adapter_state_dict(
            model.backbone, checkpoint["ema_adapters" if ema else "adapters"]
        )


def _amp(device: torch.device | str):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda")
        else nullcontext()
    )


def training_views(manifest: dict) -> tuple[list[dict], list[dict]]:
    names = [str(view["name"]) for view in manifest["views"]]
    if len(names) != len(set(names)):
        raise ValueError("Manifest view names must be unique")
    train = [view for view in manifest["views"] if view["split"] == "train"]
    labeled = [view for view in train if view.get("mask_path")]
    unlabeled = [view for view in train if not view.get("mask_path")]
    if not labeled:
        raise ValueError("Teacher training requires at least one labeled training view")
    return labeled, unlabeled


def verify_pseudo_provenance(
    manifest_path: str | Path, pseudo_dir: str | Path, *, verify_files: bool = True
) -> dict:
    """Reject incomplete exports, different data splits and unexpected training views.

    Consumers should call this once before loading any pseudo targets. ``verify_files``
    additionally validates bytes, native grids and the provenance embedded in each NPZ.
    """
    manifest_path, pseudo_dir = Path(manifest_path).resolve(), Path(pseudo_dir).resolve()
    manifest = json.loads(manifest_path.read_text())
    provenance = json.loads((pseudo_dir / "provenance.json").read_text())
    manifest_hash = file_sha256(manifest_path)
    if not provenance.get("complete"):
        raise ValueError("Pseudo export is incomplete")
    if provenance.get("manifest_sha256") != manifest_hash:
        raise ValueError("Pseudo manifest differs from the current training/validation split")
    if (
        provenance.get("kind") != "train_only_teacher_soft_probabilities"
        or provenance.get("split") != "train"
    ):
        raise ValueError("Pseudo export is not explicitly train-only")
    if provenance.get("class_names") != manifest["class_names"]:
        raise ValueError("Pseudo class ordering differs from the manifest")
    source = provenance["teacher_provenance"]
    require_matching_protocol(provenance, manifest, "pseudo directory and manifest")
    require_teacher_manifest_protocol(source, manifest, "pseudo teacher and manifest")
    if source.get("manifest_sha256") != manifest_hash:
        raise ValueError("Teacher was trained under a different manifest")
    allowed = {str(view["name"]): view for view in manifest["views"] if view["split"] == "train"}
    seen = set()
    for entry in provenance["views"]:
        name = str(entry["name"])
        if name not in allowed or name in seen or entry.get("split") != "train":
            raise ValueError(f"Invalid or duplicate pseudo training view: {name}")
        seen.add(name)
        view = allowed[name]
        path = pseudo_dir / (Path(name).stem + ".npz")
        if (entry["height"], entry["width"]) != (view["height"], view["width"]):
            raise ValueError(f"Pseudo native grid differs: {name}")
        if verify_files:
            if file_sha256(path) != entry["sha256"]:
                raise ValueError(f"Pseudo file checksum differs: {name}")
            with np.load(path, allow_pickle=False) as values:
                if "pixel_protocol" in values:
                    require_matching_protocol(str(values["pixel_protocol"].item()), provenance,
                                              "pseudo embedded profile")
                if (
                    str(values["split"].item()) != "train"
                    or str(values["view_name"].item()) != name
                ):
                    raise ValueError(f"Pseudo embedded view/split differs: {name}")
                if (
                    str(values["manifest_sha256"].item()) != manifest_hash
                    or str(values["checkpoint_sha256"].item()) != provenance["checkpoint_sha256"]
                ):
                    raise ValueError(f"Pseudo embedded provenance differs: {name}")
                expected = len(manifest["class_names"]), view["height"], view["width"]
                if (
                    values["probs"].shape != expected
                    or values["confidence"].shape != expected[1:]
                    or values["valid"].shape != expected[1:]
                ):
                    raise ValueError(f"Pseudo tensor shape differs: {name}")
    actual_names = {path.name for path in pseudo_dir.glob("*.npz")}
    expected_names = {Path(name).stem + ".npz" for name in seen}
    if actual_names != expected_names:
        raise ValueError("Pseudo directory contains missing or unregistered files")
    optimized = set(source["train_labeled_views"]) | set(source["train_unlabeled_views"])
    if not optimized.issubset(allowed):
        raise ValueError("Teacher optimization provenance includes held-out views")
    if record := provenance.get("train_confidence_calibration"):
        calibration_path = pseudo_dir / "train_confidence_calibration.json"
        if file_sha256(calibration_path) != record["sha256"]:
            raise ValueError("Training confidence calibration checksum differs")
        calibration = json.loads(calibration_path.read_text())
        require_matching_protocol(calibration, provenance, "pseudo confidence calibration")
        labeled_names = set(calibration["labeled_views"])
        if (
            calibration["split"] != "train"
            or calibration["manifest_sha256"] != manifest_hash
            or calibration["checkpoint_sha256"] != provenance["checkpoint_sha256"]
            or set(calibration["prediction_views"]) != seen
            or not labeled_names.issubset(seen)
            or not labeled_names.issubset({name for name, view in allowed.items() if view.get("mask_path")})
        ):
            raise ValueError("Confidence calibration includes invalid source views")
    return provenance


class ViewReader:
    def __init__(self, classes: int, cache_size: int = 8):
        self.classes = classes
        self.cache_size = cache_size
        self.cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def read(self, view: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Real and rendered RGB can share a camera name but never cached pixels.
        key = (str(view["image_path"]), view.get("mask_path"), view.get("valid_path"))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        image = cv2.imread(str(view["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(view["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        if (h, w) != (view["height"], view["width"]):
            raise ValueError(f"Image grid disagrees with manifest: {key}")
        mask = np.full((h, w), IGNORE_LABEL, dtype=np.uint8)
        if view.get("mask_path"):
            mask = cv2.imread(str(view["mask_path"]), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise FileNotFoundError(view["mask_path"])
            if mask.ndim != 2 or mask.shape != (h, w):
                raise ValueError(f"Expected a class-index mask on image grid: {key}")
            if np.any((mask >= self.classes) & (mask != IGNORE_LABEL)):
                raise ValueError(f"Mask has out-of-range class labels: {key}")
        valid = np.ones((h, w), dtype=np.bool_)
        if view.get("valid_path"):
            valid_image = cv2.imread(str(view["valid_path"]), cv2.IMREAD_UNCHANGED)
            if valid_image is None or valid_image.shape != (h, w):
                raise ValueError(f"Invalid valid-pixel mask for {key}")
            valid = valid_image > 0
        mask = mask.copy()
        mask[~valid] = IGNORE_LABEL
        result = image, mask, valid
        self.cache[key] = result
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result


def aligned_crop(
    image: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
    size: int,
    rng: np.random.Generator,
    *,
    augment: bool = True,
    pixel_protocol: str = "legacy_mixed_v1",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One spatial transform for RGB/labels/valid; photometric branches follow it."""
    if augment:
        scale = float(rng.uniform(0.8, 1.2))
        h, w = image.shape[:2]
        size_wh = max(1, round(w * scale)), max(1, round(h * scale))
        image = cv2.resize(image, size_wh, interpolation=cv2.INTER_LINEAR)
        from .coordinates import resize_discrete_numpy
        mask = resize_discrete_numpy(mask, size_wh, pixel_protocol)
        valid = resize_discrete_numpy(valid.astype(np.uint8), size_wh, pixel_protocol) > 0
    h, w = image.shape[:2]
    pad_h, pad_w = max(0, size - h), max(0, size - w)
    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=IGNORE_LABEL)
    valid = np.pad(valid, ((0, pad_h), (0, pad_w)), constant_values=False)
    h, w = image.shape[:2]
    y, x = int(rng.integers(h - size + 1)), int(rng.integers(w - size + 1))
    classes = np.unique(mask[(mask > 0) & (mask != IGNORE_LABEL)])
    if augment and len(classes) and rng.random() < 0.5:
        # Equal class choice reduces loss of foundations and small foreground regions.
        selected = rng.choice(classes)
        coordinates = np.argwhere(mask == selected)
        cy, cx = coordinates[int(rng.integers(len(coordinates)))]
        y = int(np.clip(cy - rng.integers(size), 0, h - size))
        x = int(np.clip(cx - rng.integers(size), 0, w - size))
    image, mask, valid = (
        image[y : y + size, x : x + size],
        mask[y : y + size, x : x + size],
        valid[y : y + size, x : x + size],
    )
    if augment and rng.random() < 0.5:
        image, mask, valid = image[:, ::-1], mask[:, ::-1], valid[:, ::-1]
    return (
        torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255,
        torch.from_numpy(mask.copy()).long(),
        torch.from_numpy(valid.copy()),
    )


def photometric_augment(image: torch.Tensor, strength: float) -> torch.Tensor:
    """Photometric only: no cutout, resampling, or flip that could misalign targets."""
    shape = (image.shape[0], 1, 1, 1)
    gain = 1 + (torch.rand(shape, device=image.device) * 2 - 1) * strength
    contrast = 1 + (torch.rand(shape, device=image.device) * 2 - 1) * strength
    saturation = 1 + (torch.rand(shape, device=image.device) * 2 - 1) * strength
    value = (image - image.mean(dim=(-2, -1), keepdim=True)) * contrast + image.mean(
        dim=(-2, -1), keepdim=True
    )
    gray = value.mean(dim=1, keepdim=True)
    value = (gray + (value - gray) * saturation) * gain
    return (value + torch.randn_like(value) * (0.025 * strength)).clamp(0, 1)


def aligned_context_frame(
    image: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
    short_side: int,
    rng: np.random.Generator,
    *,
    augment: bool = True,
    patch_size: int = 16,
    pixel_protocol: str = "legacy_mixed_v1",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preserve a whole camera frame and its aspect ratio on the ViT patch grid.

    Unlike crop sampling, this retains tower/deck/foundation context. Padding is
    excluded from both supervised and consistency losses, and all spatial
    operations are shared by RGB, labels, and valid pixels.
    """
    h, w = image.shape[:2]
    scale = short_side / min(h, w)
    target_h, target_w = max(1, round(h * scale)), max(1, round(w * scale))
    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    from .coordinates import resize_discrete_numpy
    mask = resize_discrete_numpy(mask, (target_w, target_h), pixel_protocol)
    valid = resize_discrete_numpy(valid.astype(np.uint8), (target_w, target_h), pixel_protocol) > 0
    mask[~valid] = IGNORE_LABEL
    if augment and rng.random() < 0.5:
        image, mask, valid = image[:, ::-1], mask[:, ::-1], valid[:, ::-1]
    pad_h, pad_w = -target_h % patch_size, -target_w % patch_size
    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=IGNORE_LABEL)
    valid = np.pad(valid, ((0, pad_h), (0, pad_w)), constant_values=False)
    return (
        torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255,
        torch.from_numpy(mask.copy()).long(),
        torch.from_numpy(valid.copy()),
    )


def semantic_edges(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = target != IGNORE_LABEL
    edges = torch.zeros_like(valid)
    horizontal = (target[:, :, 1:] != target[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    vertical = (target[:, 1:, :] != target[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
    edges[:, :, 1:] |= horizontal
    edges[:, :, :-1] |= horizontal
    edges[:, 1:, :] |= vertical
    edges[:, :-1, :] |= vertical
    # Exclude boundaries adjacent to invalid/ignored pixels from boundary supervision.
    trusted = F.max_pool2d((~valid).float().unsqueeze(1), 3, 1, 1).squeeze(1) == 0
    return edges, valid & trusted


def supervised_loss(
    output: dict[str, torch.Tensor], target: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = output["logits"].float()
    valid = target != IGNORE_LABEL
    edges, edge_valid = semantic_edges(target)
    ce = F.cross_entropy(logits, target, ignore_index=IGNORE_LABEL, reduction="none")
    weight = valid.float() * (1 + 2 * edges.float())
    ce = (ce * weight).sum() / weight.sum().clamp_min(1)
    classes = logits.shape[1]
    one_hot = F.one_hot(target.clamp(0, classes - 1), classes).permute(
        0, 3, 1, 2
    ).float() * valid.unsqueeze(1)
    probs = logits.softmax(1) * valid.unsqueeze(1)
    numerator = 2 * (probs * one_hot).sum((0, 2, 3))
    denominator = (probs + one_hot).sum((0, 2, 3))
    present = one_hot.sum((0, 2, 3)) > 0
    dice = (
        (1 - (numerator[present] + 1) / (denominator[present] + 1)).mean()
        if present.any()
        else logits.sum() * 0
    )
    boundary_logits = output["boundary_logits"].float().squeeze(1)
    edge_weight = edge_valid.float() * (1 + 4 * edges.float())
    boundary = (
        F.binary_cross_entropy_with_logits(boundary_logits, edges.float(), reduction="none")
        * edge_weight
    ).sum() / edge_weight.sum().clamp_min(1)
    total = ce + 0.5 * dice + 0.1 * boundary
    return total, {
        "ce": float(ce.detach()),
        "dice": float(dice.detach()),
        "boundary": float(boundary.detach()),
    }


@torch.no_grad()
def update_ema(ema: nn.Module, student: nn.Module, decay: float) -> None:
    for target, source in zip(ema.parameters(), student.parameters(), strict=True):
        target.lerp_(source, 1 - decay)
    for target, source in zip(ema.buffers(), student.buffers(), strict=True):
        target.copy_(source)


class ClassConfidence:
    """Train-label-calibrated per-class acceptance; no validation labels are used."""

    def __init__(self, classes: int, device: torch.device | str):
        self.thresholds = torch.full((classes,), 0.95, device=device)

    @torch.no_grad()
    def update(self, probabilities: torch.Tensor, target: torch.Tensor) -> None:
        confidence, prediction = probabilities.max(1)
        for index in range(len(self.thresholds)):
            values = confidence[(target == index) & (prediction == index)]
            if values.numel() >= 16:
                value = torch.quantile(values.float(), 0.6).clamp(0.6, 0.95)
                self.thresholds[index].lerp_(value, 0.01)

    def loss(
        self, logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        confidence, prediction = targets.max(1)
        accepted = valid & (confidence >= self.thresholds[prediction])
        entropy = -(targets * targets.clamp_min(1e-8).log()).sum(1) / math.log(targets.shape[1])
        weights = accepted.float() * (1 - entropy).clamp(0, 1)
        pixel_loss = -(targets.detach() * logits.float().log_softmax(1)).sum(1)
        # Normalize by valid area so a handful of accepted pixels cannot dominate.
        loss = (pixel_loss * weights).sum() / valid.sum().clamp_min(1)
        return loss, float(accepted.sum() / valid.sum().clamp_min(1))


def _tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    return sorted(set(list(range(0, length - tile + 1, stride)) + [length - tile]))


@torch.inference_mode()
def predict_context_image(
    model: DINOv3Teacher,
    image: np.ndarray,
    *,
    decoder: nn.Module | None = None,
    short_side: int = 768,
    flip: bool = False,
) -> np.ndarray:
    """Predict one aspect-preserving whole image without consulting labels."""
    h, w = image.shape[:2]
    scale = short_side / min(h, w)
    target_h, target_w = max(1, round(h * scale)), max(1, round(w * scale))
    frame = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    frame = np.pad(
        frame,
        ((0, -target_h % model.patch_size), (0, -target_w % model.patch_size), (0, 0)),
        mode="edge",
    )
    device = next(model.decoder.parameters()).device
    tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255
    model.eval()
    if decoder is not None:
        decoder.eval()
    with _amp(device):
        probability = model(tensor, decoder=decoder)["logits"].float().softmax(1)
        if flip:
            second = model(tensor.flip(-1), decoder=decoder)["logits"].float().softmax(1).flip(-1)
            probability = (probability + second) * 0.5
    probability = F.interpolate(
        probability[:, :, :target_h, :target_w], size=(h, w), mode="bilinear", align_corners=False
    )
    return probability[0].cpu().numpy()


@torch.inference_mode()
def predict_image(
    model: DINOv3Teacher,
    image: np.ndarray,
    *,
    decoder: nn.Module | None = None,
    tile_size: int = 768,
    stride: int = 512,
    flip: bool = True,
    context_weight: float = 0.0,
    context_short_side: int = 768,
) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping native-grid probability blending; returns CHW probabilities and HW confidence."""
    if tile_size % model.patch_size or stride <= 0 or stride > tile_size:
        raise ValueError("Tile size must be patch-aligned, and 0 < stride <= tile size")
    if not 0 <= context_weight <= 1 or context_short_side < 32:
        raise ValueError("Invalid context inference weight or short side")
    h, w = image.shape[:2]
    padded = np.pad(
        image, ((0, max(0, tile_size - h)), (0, max(0, tile_size - w)), (0, 0)), mode="edge"
    )
    ph, pw = padded.shape[:2]
    device = next(model.decoder.parameters()).device
    classes = model.decoder.head[-1].out_channels
    accumulator = np.zeros((classes, ph, pw), dtype=np.float32)
    weight_sum = np.zeros((ph, pw), dtype=np.float32)
    disagreement_sum = np.zeros((ph, pw), dtype=np.float32)
    coordinates = np.linspace(-1, 1, tile_size, dtype=np.float32)
    kernel = np.exp(-2 * (coordinates[:, None] ** 2 + coordinates[None, :] ** 2))
    model.eval()
    if decoder is not None:
        decoder.eval()
    for y in _tile_starts(ph, tile_size, stride):
        for x in _tile_starts(pw, tile_size, stride):
            tensor = (
                torch.from_numpy(padded[y : y + tile_size, x : x + tile_size].copy())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
                .float()
                / 255
            )
            with _amp(device):
                base = model(tensor, decoder=decoder)["logits"].float().softmax(1)
                if flip:
                    second = (
                        model(tensor.flip(-1), decoder=decoder)["logits"]
                        .float()
                        .softmax(1)
                        .flip(-1)
                    )
                    average = (base + second) * 0.5
                    js = (
                        0.5
                        * (
                            (
                                base * (base.clamp_min(1e-8).log() - average.clamp_min(1e-8).log())
                            ).sum(1)
                            + (
                                second
                                * (second.clamp_min(1e-8).log() - average.clamp_min(1e-8).log())
                            ).sum(1)
                        )
                        / math.log(2)
                    )
                else:
                    average, js = base, torch.zeros_like(base[:, 0])
            accumulator[:, y : y + tile_size, x : x + tile_size] += (
                average[0].cpu().numpy() * kernel
            )
            weight_sum[y : y + tile_size, x : x + tile_size] += kernel
            disagreement_sum[y : y + tile_size, x : x + tile_size] += js[0].cpu().numpy() * kernel
    probabilities = (accumulator / weight_sum[None])[:, :h, :w]
    disagreement = (disagreement_sum / weight_sum)[:h, :w].clip(0, 1)
    if context_weight > 0:
        context = predict_context_image(
            model, image, decoder=decoder, short_side=context_short_side, flip=flip
        )
        midpoint = (probabilities + context) * 0.5
        scale_js = 0.5 * (
            (probabilities * np.log(probabilities.clip(1e-8) / midpoint.clip(1e-8))).sum(0)
            + (context * np.log(context.clip(1e-8) / midpoint.clip(1e-8))).sum(0)
        ) / math.log(2)
        disagreement = np.clip(
            (1 - context_weight) * disagreement
            + 4 * context_weight * (1 - context_weight) * scale_js,
            0, 1,
        )
        probabilities = (1 - context_weight) * probabilities + context_weight * context
    probabilities /= probabilities.sum(0, keepdims=True).clip(1e-8)
    entropy = -(probabilities * np.log(probabilities.clip(1e-8))).sum(0) / math.log(classes)
    confidence = (probabilities.max(0) * (1 - entropy).clip(0, 1) * (1 - disagreement)).astype(
        np.float32
    )
    return probabilities, confidence


def confusion_metrics(confusion: np.ndarray) -> dict:
    union = confusion.sum(0) + confusion.sum(1) - np.diag(confusion)
    present = union > 0
    ious = np.divide(
        np.diag(confusion), union, out=np.zeros(len(union), dtype=float), where=present
    )
    return {
        "miou": float(ious[present].mean()) if present.any() else None,
        "per_class_iou": [float(v) if p else None for v, p in zip(ious, present)],
        "pixel_accuracy": float(np.trace(confusion) / confusion.sum()) if confusion.sum() else None,
        "confusion": confusion.tolist(),
    }


def confidence_calibration_summary(
    total_confusion: np.ndarray,
    accepted_confusions: dict[float, np.ndarray],
    all_predicted: np.ndarray,
    all_accepted_predicted: dict[float, np.ndarray],
) -> dict:
    """Train-label-only acceptance/error statistics on the exported FP16 targets."""
    def ratio(a, b):
        return [float(x / y) if y else None for x, y in zip(a, b, strict=True)]

    gt_support, pred_support = total_confusion.sum(1), total_confusion.sum(0)
    result = {}
    for threshold, matrix in accepted_confusions.items():
        accepted_predicted, accepted_gt = matrix.sum(0), matrix.sum(1)
        correct = np.diag(matrix)
        precision = ratio(correct, accepted_predicted)
        result[str(threshold)] = {
            "labeled_train_confusion": matrix.tolist(),
            "labeled_train_predicted_class_coverage": ratio(accepted_predicted, pred_support),
            "labeled_train_prediction_error_rate": [1 - value if value is not None else None for value in precision],
            "labeled_train_target_class_coverage": ratio(accepted_gt, gt_support),
            "labeled_train_correct_target_class_coverage": ratio(correct, gt_support),
            "labeled_train_accepted_predicted_counts": accepted_predicted.tolist(),
            "all_train_predicted_class_coverage": ratio(all_accepted_predicted[threshold], all_predicted),
            "all_train_accepted_predicted_counts": all_accepted_predicted[threshold].tolist(),
        }
    return {
        "labeled_train_full_confusion": total_confusion.tolist(),
        "labeled_train_target_class_support": gt_support.tolist(),
        "labeled_train_predicted_class_support": pred_support.tolist(),
        "all_train_predicted_class_support": all_predicted.tolist(),
        "threshold_metrics": result,
        "note": "Train-fit reliability statistics; validation labels were not read and these errors do not guarantee unseen-view error rates",
    }


@torch.inference_mode()
def evaluate_teacher(
    model: DINOv3Teacher,
    views: list[dict],
    reader: ViewReader,
    *,
    decoder: nn.Module | None,
    tile_size: int,
    stride: int,
    flip: bool = False,
    context_weight: float = 0.0,
    context_short_side: int = 768,
) -> dict:
    count = reader.classes
    confusion = np.zeros((count, count), dtype=np.int64)
    for view in views:
        image, mask, valid = reader.read(view)
        probabilities, _ = predict_image(
            model, image, decoder=decoder, tile_size=tile_size, stride=stride, flip=flip,
            context_weight=context_weight, context_short_side=context_short_side,
        )
        keep = valid & (mask != IGNORE_LABEL)
        encoded = mask[keep].astype(np.int64) * count + probabilities.argmax(0)[keep]
        confusion += np.bincount(encoded, minlength=count * count).reshape(count, count)
    return {
        **confusion_metrics(confusion),
        "views": [v["name"] for v in views],
        "split": "val",
        "labeled_views": [v["name"] for v in views if v.get("mask_path")],
        "image_domains": sorted({v.get("image_domain", "real_rgb") for v in views}),
        "flip_tta": flip,
        "context_weight": context_weight,
        "context_short_side": context_short_side,
    }


@dataclass
class TeacherConfig:
    steps: int = 6000
    crop_size: int = 768
    channels: int = 192
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    consistency_start: int = 1000
    consistency_weight: float = 0.5
    ema_decay: float = 0.995
    eval_every: int = 500
    log_every: int = 25
    seed: int = 20260926
    device: str = "cuda"
    val_limit: int = 0
    val_stride: int = 512
    gradient_clip: float = 1.0
    context_start: int = 3000
    context_probability: float = 0.0
    context_short_side: int = 768
    cpu_threads: int = 8
    adapter_rank: int = 0
    adapter_blocks: int = 4
    adapter_alpha: float = 16.0
    adapter_lr: float = 1e-5
    warmstart_checkpoint: str = ""
    render_mix_probability: float = 0.0
    eval_flip: bool = False
    eval_context_weight: float = 0.0
    eval_context_short_side: int = 768
    eval_all_validation_views: bool = False
    independent_augmentation_rng: bool = False
    checkpoint_every: int = 0
    evaluate_validation: bool = True


def train_teacher(
    manifest_path: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    config: TeacherConfig | None = None,
    *,
    resume: str | Path | None = None,
) -> dict:
    config = config or TeacherConfig()
    if config.steps < 1 or config.crop_size % 16 or config.crop_size < 32:
        raise ValueError("Use positive steps and a crop size divisible by 16, at least 32")
    if config.channels < 16 or config.channels % 16:
        raise ValueError("Decoder channels must be a positive multiple of 16, at least 16")
    if config.eval_every < 1 or config.log_every < 1 or not 0 <= config.ema_decay < 1:
        raise ValueError("Invalid evaluation/log intervals or EMA decay")
    if config.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be nonnegative; zero keeps evaluation-only saves")
    if not isinstance(config.evaluate_validation, bool):
        raise TypeError("evaluate_validation must be an explicit boolean")
    if not 0 <= config.context_probability <= 1 or config.context_short_side < 32:
        raise ValueError("Invalid context probability or short side")
    if config.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    if config.adapter_rank < 0 or config.adapter_blocks < 1 or config.adapter_alpha <= 0:
        raise ValueError("Invalid adapter dimensions or scaling")
    if config.lr <= 0 or config.adapter_lr <= 0:
        raise ValueError("Learning rates must be positive")
    if not 0 <= config.render_mix_probability <= 1 or not 0 <= config.eval_context_weight <= 1:
        raise ValueError("Invalid render mixing or evaluation context weight")
    torch.set_num_threads(config.cpu_threads)
    manifest_path, model_dir, output_dir = (
        Path(manifest_path).resolve(),
        Path(model_dir).resolve(),
        Path(output_dir).resolve(),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    from .teacher_domains import domain_schedule, select_image_source, validate_image_sources

    training_pixel_protocol = pixel_protocol(manifest)
    image_source_protocol = validate_image_sources(manifest, verify_validation_files=config.evaluate_validation)
    verify_teacher_render_protocol(manifest, image_source_protocol)
    manifest = annotate_manifest(manifest)
    if config.render_mix_probability and not image_source_protocol:
        raise ValueError("Render mixing needs an audited derived manifest")
    if image_source_protocol and config.consistency_start <= config.steps:
        raise ValueError("Rendered-domain adaptation currently requires supervised-only training")
    schedule = domain_schedule(config.steps, config.render_mix_probability, config.seed)
    labeled, unlabeled = training_views(manifest)
    val_views = [v for v in manifest["views"] if v["split"] == "val" and (
        config.eval_all_validation_views or v.get("mask_path")
    )]
    if config.val_limit:
        val_views = val_views[: config.val_limit]
    if not config.evaluate_validation:
        val_views = []
    classes = len(manifest["class_names"])
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)
    model = load_teacher(
        model_dir, classes, config.channels, str(device),
        pixel_profile=training_pixel_protocol,
        **checkpoint_adapter_options(asdict(config)),
    )
    ema = deepcopy(model.decoder).eval().requires_grad_(False)
    trainable = list(model.decoder.parameters())
    parameter_groups = [{"params": trainable, "lr": config.lr}]
    ema_adapters = {}
    if config.adapter_rank:
        from .teacher_adapters import (
            adapter_named_parameters,
            adapter_state_dict,
            temporary_adapter_state,
            update_adapter_ema,
        )

        adapter_parameters = [p for _, p in adapter_named_parameters(model.backbone)]
        trainable = trainable + adapter_parameters
        parameter_groups.append({"params": adapter_parameters, "lr": config.adapter_lr})
        ema_adapters = adapter_state_dict(model.backbone)
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config.weight_decay)
    confidence = ClassConfidence(classes, device)
    reader = ViewReader(classes)
    manifest_hash = file_sha256(manifest_path)
    provenance = {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "pixel_protocol": protocol_metadata(training_pixel_protocol),
        "model_dir": str(model_dir),
        "model_config_sha256": file_sha256(model_dir / "config.json"),
        "model_weights_sha256": {
            path.name: file_sha256(path) for path in sorted(model_dir.glob("*.safetensors"))
        },
        "model_index_sha256": backbone_index_hashes(model_dir),
        "class_names": manifest["class_names"],
        "train_labeled_views": [v["name"] for v in labeled],
        "train_unlabeled_views": [v["name"] for v in unlabeled],
        "validation_views": [v["name"] for v in val_views],
        "validation_policy": ("Only held-out metric evaluation and checkpoint selection; excluded from optimization and pseudo export"
                              if config.evaluate_validation else "Disabled: no VAL pixel reads/evaluation/selection; save fixed last EMA"),
        "configuration": asdict(config),
        "frozen_backbone": not bool(config.adapter_rank),
        "base_backbone_frozen": True,
        "trainable_adapter_parameters": sum(
            p.numel() for p in model.backbone.parameters() if p.requires_grad
        ),
        "source_sha256": file_sha256(Path(__file__)),
    }
    if image_source_protocol:
        provenance["image_source_protocol"] = image_source_protocol
        provenance["domain_schedule"] = {
            "real": int((schedule == 0).sum()), "rendered": int(schedule.sum()),
            "sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
            "policy": "Predeclared exact-count shuffled schedule, independent of view/crop RNG",
        }
    if (model_dir / "download_provenance.json").exists():
        provenance["backbone_source"] = json.loads(
            (model_dir / "download_provenance.json").read_text()
        )
    start_step, best_miou = 0, -1.0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        require_teacher_manifest_protocol(checkpoint, manifest, "teacher resume")
        if checkpoint_adapter_options(checkpoint["configuration"]) != checkpoint_adapter_options(asdict(config)):
            raise ValueError("Resume adapter configuration differs")
        if checkpoint["provenance"]["manifest_sha256"] != manifest_hash:
            raise ValueError("Resume manifest differs; refusing split/label leakage")
        if image_source_protocol and checkpoint["provenance"]["domain_schedule"] != provenance["domain_schedule"]:
            raise ValueError("Resume image domain schedule differs")
        if checkpoint["provenance"]["model_config_sha256"] != provenance["model_config_sha256"]:
            raise ValueError("Resume backbone configuration differs")
        if (
            checkpoint["provenance"].get("model_weights_sha256")
            != provenance["model_weights_sha256"]
        ):
            raise ValueError("Resume backbone weights differ")
        require_backbone_index(checkpoint["provenance"], provenance["model_index_sha256"], "teacher resume")
        if checkpoint["configuration"].get("independent_augmentation_rng", False) != config.independent_augmentation_rng:
            raise ValueError("Resume augmentation RNG policy differs")
        if checkpoint["configuration"].get("evaluate_validation", True) != config.evaluate_validation:
            raise ValueError("Resume validation policy differs")
        model.decoder.load_state_dict(checkpoint["decoder"])
        ema.load_state_dict(checkpoint["ema_decoder"])
        load_checkpoint_adapters(model, checkpoint, ema=False)
        if config.adapter_rank:
            ema_adapters = {key: value.to(device) for key, value in checkpoint["ema_adapters"].items()}
        optimizer.load_state_dict(checkpoint["optimizer"])
        confidence.thresholds.copy_(checkpoint["class_thresholds"])
        start_step, best_miou = checkpoint["step"], checkpoint["best_miou"]
        rng.bit_generator.state = checkpoint["numpy_generator_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in checkpoint["cuda_rng_state"]])
        if start_step >= config.steps:
            raise ValueError("Resume step must be smaller than requested total steps")
        provenance["resume"] = {
            "checkpoint": str(Path(resume).resolve()),
            "sha256": file_sha256(resume),
            "step": start_step,
            "configuration": checkpoint["configuration"],
            "pixel_protocol": protocol_metadata(checkpoint_pixel_protocol(checkpoint)),
        }
    elif (output_dir / "last.pt").exists():
        raise FileExistsError(
            "Output already has a checkpoint; use --resume or a new output directory"
        )
    if config.warmstart_checkpoint and not resume:
        initial = torch.load(config.warmstart_checkpoint, map_location=device, weights_only=False)
        require_teacher_manifest_protocol(initial, manifest, "teacher warmstart")
        if initial["configuration"].get("adapter_rank", 0) and (
            checkpoint_adapter_options(initial["configuration"])
            != checkpoint_adapter_options(asdict(config))
        ):
            raise ValueError("Warmstart adapter configuration differs")
        allowed_manifest_hashes = {manifest_hash}
        if image_source_protocol:
            allowed_manifest_hashes.add(image_source_protocol["original_manifest_sha256"])
        common_original = None
        if initial["provenance"]["manifest_sha256"] not in allowed_manifest_hashes:
            from .teacher_domains import MULTI_COMPONENT_DOMAIN, verify_common_original_warmstart

            if not image_source_protocol or image_source_protocol.get("id") != MULTI_COMPONENT_DOMAIN:
                raise ValueError("Warmstart manifest differs from verified split lineage")
            common_original = verify_common_original_warmstart(initial, image_source_protocol)
        for key in ("model_config_sha256", "model_weights_sha256"):
            if initial["provenance"][key] != provenance[key]:
                raise ValueError(f"Warmstart {key} differs")
        require_backbone_index(initial["provenance"], provenance["model_index_sha256"], "teacher warmstart")
        model.decoder.load_state_dict(initial["ema_decoder"])
        ema.load_state_dict(initial["ema_decoder"])
        confidence.thresholds.copy_(initial["class_thresholds"])
        if config.adapter_rank and initial.get("ema_adapters"):
            load_checkpoint_adapters(model, initial)
            ema_adapters = adapter_state_dict(model.backbone)
        provenance["warmstart"] = {
            "checkpoint": str(Path(config.warmstart_checkpoint).resolve()),
            "sha256": file_sha256(config.warmstart_checkpoint),
            "step": initial["step"],
            "configuration": initial["configuration"],
            "pixel_protocol": protocol_metadata(checkpoint_pixel_protocol(initial)),
            "policy": "EMA weights only; fresh optimizer, step count, and RNG",
        }
        if common_original is not None:
            provenance["warmstart"]["common_original_lineage"] = common_original
    if not resume and (config.warmstart_checkpoint or config.independent_augmentation_rng):
        # Different backbone/head sizes must not shift photometric augmentation
        # RNG in a matched capacity comparison. Resume restores its saved state.
        torch.manual_seed(config.seed + 1)
    _write_json(output_dir / "provenance.json", provenance)
    start_time = time.monotonic()
    last_validation = None

    def save_checkpoint(step: int, destination: str) -> None:
        value = {
            "format_version": 1,
            "pixel_protocol": protocol_metadata(training_pixel_protocol),
            "step": step,
            "best_miou": best_miou,
            "decoder": model.decoder.state_dict(),
            "ema_decoder": ema.state_dict(),
            "adapters": adapter_state_dict(model.backbone) if config.adapter_rank else {},
            "ema_adapters": ema_adapters,
            "optimizer": optimizer.state_dict(),
            "class_thresholds": confidence.thresholds,
            "configuration": asdict(config),
            "provenance": provenance,
            "numpy_generator_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "validation": last_validation,
            "domain_counts": {
                "real": int((schedule[:step] == 0).sum()),
                "rendered": int(schedule[:step].sum()),
            },
        }
        temporary = output_dir / (destination + ".tmp")
        torch.save(value, temporary)
        temporary.replace(output_dir / destination)

    for index in range(start_step, config.steps):
        step = index + 1
        model.train()
        view = labeled[int(rng.integers(len(labeled)))]
        image_domain = "real"
        if image_source_protocol:
            view = select_image_source(view, bool(schedule[index]))
            image_domain = view["image_domain"]
        context_frame = (
            config.context_probability > 0
            and step >= config.context_start
            and rng.random() < config.context_probability
        )
        if context_frame:
            image, labels, _ = aligned_context_frame(
                *reader.read(view), config.context_short_side, rng,
                pixel_protocol=training_pixel_protocol,
            )
        else:
            image, labels, _ = aligned_crop(*reader.read(view), config.crop_size, rng,
                                           pixel_protocol=training_pixel_protocol)
        image, labels = image.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)
        learning_rate = (
            config.lr
            * min(1.0, step / max(1, config.warmup_steps))
            * max(0.01, (1 - index / config.steps) ** 0.9)
        )
        for group_index, group in enumerate(optimizer.param_groups):
            group["lr"] = learning_rate * (config.adapter_lr / config.lr if group_index else 1)
        optimizer.zero_grad(set_to_none=True)
        with _amp(device):
            labeled_augmented = photometric_augment(image, 0.25)
            labeled_features = model.extract_features(labeled_augmented)
            output = model.decoder(labeled_features, labeled_augmented)
            loss, components = supervised_loss(output, labels)
        # Backward separately releases the labeled graph before the unlabeled graph.
        loss.backward()
        total_loss = float(loss.detach())
        accepted = 0.0
        if unlabeled and step >= config.consistency_start:
            with torch.no_grad(), _amp(device):
                if config.adapter_rank:
                    with temporary_adapter_state(model.backbone, ema_adapters):
                        calibrated = model(labeled_augmented, decoder=ema)["logits"].float().softmax(1)
                else:
                    calibrated = ema(labeled_features, labeled_augmented)["logits"].float().softmax(1)
                confidence.update(calibrated, labels)
            un_view = unlabeled[int(rng.integers(len(unlabeled)))]
            un_image, _, un_valid = aligned_crop(*reader.read(un_view), config.crop_size, rng,
                                                pixel_protocol=training_pixel_protocol)
            un_image, un_valid = un_image.unsqueeze(0).to(device), un_valid.unsqueeze(0).to(device)
            adapter_context = (
                temporary_adapter_state(model.backbone, ema_adapters)
                if config.adapter_rank else nullcontext()
            )
            with torch.no_grad(), _amp(device), adapter_context:
                targets = (
                    model(photometric_augment(un_image, 0.05), decoder=ema)["logits"]
                    .float()
                    .softmax(1)
                )
            with _amp(device):
                un_output = model(photometric_augment(un_image, 0.5))["logits"]
                consistency, accepted = confidence.loss(un_output, targets, un_valid)
                ramp = min(1.0, (step - config.consistency_start + 1) / max(1, config.warmup_steps))
                unsupervised_loss = config.consistency_weight * ramp * consistency
            unsupervised_loss.backward()
            total_loss += float(unsupervised_loss.detach())
            components["consistency"] = float(consistency.detach())
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, config.gradient_clip
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                f"Non-finite decoder gradient at step {step}; last.pt retained"
            )
        optimizer.step()
        update_ema(ema, model.decoder, min(config.ema_decay, 1 - 1 / (step + 1)))
        if config.adapter_rank:
            update_adapter_ema(ema_adapters, model.backbone, min(config.ema_decay, 1 - 1 / (step + 1)))
        if step % config.log_every == 0 or step == 1:
            record = {
                "step": step,
                "loss": total_loss,
                **components,
                "accepted_pseudo_fraction": accepted,
                "class_thresholds": confidence.thresholds.detach().cpu().tolist(),
                "lr": learning_rate,
                "gradient_norm": float(gradient_norm),
                "context_frame": context_frame,
                "input_shape": list(image.shape[-2:]),
                "image_domain": image_domain,
                "image_name": view["name"],
                "image_path": view["image_path"],
                "domain_counts": {
                    "real": int((schedule[:step] == 0).sum()),
                    "rendered": int(schedule[:step].sum()),
                },
                "elapsed_seconds": time.monotonic() - start_time,
            }
            with (output_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record), flush=True)
        if step % config.eval_every == 0 or step == config.steps:
            if val_views:
                with temporary_adapter_state(model.backbone, ema_adapters) if config.adapter_rank else nullcontext():
                    last_validation = evaluate_teacher(
                        model,
                        val_views,
                        reader,
                        decoder=ema,
                        tile_size=config.crop_size,
                        stride=min(config.val_stride, config.crop_size),
                        flip=config.eval_flip,
                        context_weight=config.eval_context_weight,
                        context_short_side=config.eval_context_short_side,
                    )
                last_validation["step"] = step
                _write_json(output_dir / f"validation_{step:06d}.json", last_validation)
                if last_validation["miou"] is not None and last_validation["miou"] > best_miou:
                    best_miou = last_validation["miou"]
                    save_checkpoint(step, "best.pt")
                print(
                    json.dumps({"step": step, "validation_miou": last_validation["miou"]}),
                    flush=True,
                )
            save_checkpoint(step, "last.pt")
        elif config.checkpoint_every and step % config.checkpoint_every == 0:
            # Fault-recovery snapshots do not read or select on validation data.
            save_checkpoint(step, "last.pt")
    result = {
        "step": config.steps,
        "last_checkpoint": str(output_dir / "last.pt"),
        "best_checkpoint": str(output_dir / "best.pt")
        if (output_dir / "best.pt").exists()
        else None,
        "best_validation_miou": best_miou if best_miou >= 0 else None,
        "elapsed_seconds": time.monotonic() - start_time,
        "validation": last_validation,
        "domain_counts": {"real": int((schedule == 0).sum()), "rendered": int(schedule.sum())},
        "note": "Training complete; downstream 3D quality requires separate held-out evaluation",
    }
    _write_json(output_dir / "result.json", result)
    return result


def predict_teacher(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    model_dir: str | Path | None = None,
    device: str = "cuda",
    tile_size: int = 768,
    stride: int = 512,
    flip: bool = True,
    unlabeled_only: bool = False,
    context_weight: float = 0.0,
    context_short_side: int = 768,
) -> dict:
    manifest_path, checkpoint_path, output_dir = (
        Path(manifest_path).resolve(),
        Path(checkpoint_path).resolve(),
        Path(output_dir).resolve(),
    )
    manifest = json.loads(manifest_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = dict(checkpoint["provenance"])
    profile = require_teacher_manifest_protocol(checkpoint, manifest, "teacher prediction")
    source["pixel_protocol"] = protocol_metadata(profile)
    if source["manifest_sha256"] != file_sha256(manifest_path):
        raise ValueError(
            "Prediction manifest differs from teacher training manifest; refusing ambiguous split provenance"
        )
    model_dir = Path(model_dir or source["model_dir"]).resolve()
    if file_sha256(model_dir / "config.json") != source["model_config_sha256"]:
        raise ValueError("Backbone configuration differs from teacher checkpoint")
    current_weights = {
        path.name: file_sha256(path) for path in sorted(model_dir.glob("*.safetensors"))
    }
    if current_weights != source.get("model_weights_sha256"):
        raise ValueError("Backbone weights differ from teacher checkpoint")
    require_backbone_index(source, backbone_index_hashes(model_dir), "teacher prediction")
    classes = len(manifest["class_names"])
    model = load_teacher(
        model_dir, classes, checkpoint["configuration"]["channels"], device,
        pixel_profile=profile,
        **checkpoint_adapter_options(checkpoint["configuration"]),
    )
    model.decoder.load_state_dict(checkpoint["ema_decoder"])
    load_checkpoint_adapters(model, checkpoint)
    reader = ViewReader(classes)
    views = [
        v
        for v in manifest["views"]
        if v["split"] == "train" and (not unlabeled_only or not v.get("mask_path"))
    ]
    stems = [Path(str(view["name"])).stem for view in views]
    if len(set(stems)) != len(stems):
        raise ValueError("View filename stems must be unique for pseudo export")
    if output_dir.exists() and any(output_dir.glob("*.npz")):
        raise FileExistsError(
            "Pseudo directory already contains predictions; use a fresh output directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = file_sha256(checkpoint_path)
    provenance = {
        "format_version": 1,
        "pixel_protocol": protocol_metadata(profile),
        "pixel_protocol_binding": "Directory profile binds every registered NPZ through its SHA256, embedded manifest SHA256 and checkpoint SHA256; per-NPZ profile tags are not required.",
        "kind": "train_only_teacher_soft_probabilities",
        "inference_source_sha256": file_sha256(Path(__file__)),
        "probability_layout": "CHW",
        "grid": "native manifest image pixels",
        "manifest_sha256": source["manifest_sha256"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "teacher_step": checkpoint["step"],
        "class_names": manifest["class_names"],
        "teacher_provenance": source,
        "split": "train",
        "tile_size": tile_size,
        "stride": stride,
        "flip_tta": flip,
        "context_weight": context_weight,
        "context_short_side": context_short_side,
        "confidence_definition": (
            "max_probability * (1 - normalized_entropy) * (1 - disagreement); "
            "disagreement=clip((1-context_weight)*tile_flip_JS + "
            "4*context_weight*(1-context_weight)*tile_context_JS,0,1); "
            "both JS values normalized by log(2)"
        ),
        "views": [],
        "complete": False,
    }
    _write_json(output_dir / "provenance.json", provenance)
    (output_dir / "teacher_inference_source.py").write_text(Path(__file__).read_text())
    thresholds = (0.65, 0.8, 0.9)
    calibration_confusion = np.zeros((classes, classes), np.int64)
    accepted_confusions = {value: np.zeros_like(calibration_confusion) for value in thresholds}
    all_predicted = np.zeros(classes, np.int64)
    all_accepted_predicted = {value: np.zeros(classes, np.int64) for value in thresholds}
    calibration_views = []
    for index, view in enumerate(views):
        image, mask, valid = reader.read(view)
        probs, confidence = predict_image(
            model, image, tile_size=tile_size, stride=stride, flip=flip,
            context_weight=context_weight, context_short_side=context_short_side,
        )
        confidence[~valid] = 0
        # Audit exactly the FP16 values downstream consumers receive.
        prediction = probs.astype(np.float16).argmax(0)
        stored_confidence = confidence.astype(np.float16).astype(np.float32)
        all_predicted += np.bincount(prediction[valid], minlength=classes)
        labeled_keep = valid & (mask != IGNORE_LABEL)
        encoded = mask.astype(np.int64) * classes + prediction
        if labeled_keep.any():
            calibration_views.append(view["name"])
            calibration_confusion += np.bincount(
                encoded[labeled_keep], minlength=classes**2
            ).reshape(classes, classes)
        for threshold in thresholds:
            accepted = stored_confidence >= threshold
            all_accepted_predicted[threshold] += np.bincount(
                prediction[valid & accepted], minlength=classes
            )
            accepted_confusions[threshold] += np.bincount(
                encoded[labeled_keep & accepted], minlength=classes**2
            ).reshape(classes, classes)
        name = Path(str(view["name"])).stem
        if name != str(view["name"]) and Path(str(view["name"])).name != str(view["name"]):
            raise ValueError("View names must not contain directory components")
        destination = output_dir / (name + ".npz")
        np.savez_compressed(
            destination,
            probs=probs.astype(np.float16),
            confidence=confidence.astype(np.float16),
            valid=valid,
            view_name=np.asarray(str(view["name"])),
            split=np.asarray("train"),
            manifest_sha256=np.asarray(source["manifest_sha256"]),
            checkpoint_sha256=np.asarray(checkpoint_hash),
        )
        provenance["views"].append(
            {
                "name": view["name"],
                "path": str(destination),
                "width": view["width"],
                "height": view["height"],
                "split": "train",
                "sha256": file_sha256(destination),
            }
        )
        _write_json(output_dir / "provenance.json", provenance)
        print(f"teacher prediction {index + 1}/{len(views)}: {view['name']}", flush=True)
    calibration = {
        **confidence_calibration_summary(
            calibration_confusion, accepted_confusions, all_predicted, all_accepted_predicted
        ),
        "manifest_sha256": source["manifest_sha256"],
        "pixel_protocol": protocol_metadata(profile),
        "checkpoint_sha256": checkpoint_hash,
        "class_names": manifest["class_names"],
        "split": "train",
        "labeled_views": calibration_views,
        "prediction_views": [view["name"] for view in views],
        "confidence_definition": provenance["confidence_definition"],
        "stored_dtype": "float16",
    }
    calibration_path = output_dir / "train_confidence_calibration.json"
    _write_json(calibration_path, calibration)
    provenance["train_confidence_calibration"] = {
        "path": str(calibration_path), "sha256": file_sha256(calibration_path),
        "thresholds": list(thresholds), "labeled_views": calibration_views,
    }
    provenance["complete"] = True
    _write_json(output_dir / "provenance.json", provenance)
    return {
        "output_dir": str(output_dir),
        "views": len(views),
        "provenance": str(output_dir / "provenance.json"),
    }


@torch.inference_mode()
def evaluate_checkpoint(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    model_dir: str | Path | None = None,
    device: str = "cuda",
    tile_size: int = 768,
    stride: int = 512,
    flip: bool = False,
    limit: int = 0,
    previews: int = 6,
    context_weight: float = 0.0,
    context_short_side: int = 768,
) -> dict:
    """Evaluate a fixed checkpoint on labeled validation views; save no pseudo targets."""
    manifest_path, checkpoint_path, output_dir = (
        Path(manifest_path).resolve(),
        Path(checkpoint_path).resolve(),
        Path(output_dir).resolve(),
    )
    manifest = json.loads(manifest_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["provenance"]
    profile = require_teacher_manifest_protocol(checkpoint, manifest, "teacher evaluation")
    if source["manifest_sha256"] != file_sha256(manifest_path):
        raise ValueError("Evaluation manifest differs from the teacher training manifest")
    model_dir = Path(model_dir or source["model_dir"]).resolve()
    if file_sha256(model_dir / "config.json") != source["model_config_sha256"]:
        raise ValueError("Evaluation backbone configuration differs from the checkpoint")
    current_weights = {
        path.name: file_sha256(path) for path in sorted(model_dir.glob("*.safetensors"))
    }
    if current_weights != source.get("model_weights_sha256"):
        raise ValueError("Evaluation backbone weights differ from the checkpoint")
    require_backbone_index(source, backbone_index_hashes(model_dir), "teacher evaluation")
    classes = len(manifest["class_names"])
    model = load_teacher(
        model_dir, classes, checkpoint["configuration"]["channels"], device,
        pixel_profile=profile,
        **checkpoint_adapter_options(checkpoint["configuration"]),
    )
    model.decoder.load_state_dict(checkpoint["ema_decoder"])
    load_checkpoint_adapters(model, checkpoint)
    views = [v for v in manifest["views"] if v["split"] == "val" and v.get("mask_path")]
    if limit:
        views = views[:limit]
    if not views:
        raise ValueError("No labeled validation views in this manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = ViewReader(classes)
    confusion = np.zeros((classes, classes), dtype=np.int64)
    per_view = []
    start = time.monotonic()
    palette = np.array(
        [[35, 42, 52], [230, 160, 35], [225, 55, 65], [70, 130, 225], [110, 200, 125]],
        dtype=np.uint8,
    )
    cable_index = next(
        (i for i, name in enumerate(manifest["class_names"]) if "cable" in name), None
    )
    boundary_counts = np.zeros(4, dtype=np.int64)
    for index, view in enumerate(views):
        image, mask, valid = reader.read(view)
        probs, _ = predict_image(
            model, image, tile_size=tile_size, stride=stride, flip=flip,
            context_weight=context_weight, context_short_side=context_short_side,
        )
        prediction = probs.argmax(0)
        keep = valid & (mask != IGNORE_LABEL)
        encoded = mask[keep].astype(np.int64) * classes + prediction[keep]
        local_confusion = np.bincount(encoded, minlength=classes * classes).reshape(
            classes, classes
        )
        confusion += local_confusion
        per_view.append({"name": view["name"], **confusion_metrics(local_confusion)})
        if cable_index is not None:
            trusted = (
                cv2.erode(
                    keep.astype(np.uint8),
                    np.ones((3, 3), np.uint8),
                    borderType=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                > 0
            )
            boundaries = []
            for labels in (prediction, mask):
                binary = ((labels == cable_index) & keep).astype(np.uint8)
                boundary = binary - cv2.erode(
                    binary, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0
                )
                boundaries.append((boundary > 0) & trusted)
            pred_edge, gt_edge = boundaries
            pred_near = cv2.dilate(pred_edge.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            gt_near = cv2.dilate(gt_edge.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            boundary_counts += [
                np.count_nonzero(pred_edge & gt_near),
                np.count_nonzero(pred_edge),
                np.count_nonzero(gt_edge & pred_near),
                np.count_nonzero(gt_edge),
            ]
        if index < previews and classes == len(palette):
            gt_rgb, pred_rgb = palette[mask.clip(0, classes - 1)], palette[prediction]
            gt_rgb[~keep], pred_rgb[~valid] = 0, 0
            panel = np.concatenate((image, gt_rgb, pred_rgb), axis=1)
            panel = cv2.resize(panel, (1980, round(panel.shape[0] * 1980 / panel.shape[1])))
            title = np.zeros((36, 1980, 3), dtype=np.uint8)
            for column, text in enumerate(
                (f"RGB {view['name']}", "Validation annotation", "DINOv3 EMA prediction")
            ):
                cv2.putText(
                    title,
                    text,
                    (column * 660 + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            panel = np.concatenate((title, panel), axis=0)
            cv2.imwrite(
                str(output_dir / (Path(str(view["name"])).stem + "_comparison.jpg")),
                cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
            )
        print(f"teacher validation {index + 1}/{len(views)}: {view['name']}", flush=True)
    precision = float(boundary_counts[0] / boundary_counts[1]) if boundary_counts[1] else None
    recall = float(boundary_counts[2] / boundary_counts[3]) if boundary_counts[3] else None
    boundary_f1 = None
    if boundary_counts[1] or boundary_counts[3]:
        boundary_f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else 0.0
        )
    result = {
        **confusion_metrics(confusion),
        "per_view": per_view,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "step": checkpoint["step"],
        "manifest_sha256": source["manifest_sha256"],
        "pixel_protocol": protocol_metadata(profile),
        "manifest_split": manifest.get("split"),
        "class_names": manifest["class_names"],
        "views": [v["name"] for v in views],
        "split": "val",
        "fixed_checkpoint_evaluation": True,
        "inference_source_sha256": file_sha256(Path(__file__)),
        "checkpoint_selection_views": source["validation_views"],
        "tile_size": tile_size,
        "stride": stride,
        "flip_tta": flip,
        "context_weight": context_weight,
        "context_short_side": context_short_side,
        "cable_boundary": {
            "precision": precision,
            "recall": recall,
            "f1": boundary_f1,
            "counts": boundary_counts.tolist(),
            "definition": "inner binary class boundary, Chebyshev tolerance 2 native pixels, invalid-neighbor exclusion",
        },
        "elapsed_seconds": time.monotonic() - start,
        "note": "Development validation; do not compare directly with historical all-label training fit",
    }
    _write_json(output_dir / "metrics.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--manifest", required=True)
    train_parser.add_argument("--model-dir", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--resume")
    for name, field in TeacherConfig.__dataclass_fields__.items():
        options = {"action": argparse.BooleanOptionalAction} if isinstance(field.default, bool) else {
            "type": type(field.default)
        }
        train_parser.add_argument("--" + name.replace("_", "-"), default=field.default, **options)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--manifest", required=True)
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--model-dir")
    predict_parser.add_argument("--device", default="cuda")
    predict_parser.add_argument("--tile-size", type=int, default=768)
    predict_parser.add_argument("--stride", type=int, default=512)
    predict_parser.add_argument("--no-flip", action="store_true")
    predict_parser.add_argument("--unlabeled-only", action="store_true")
    predict_parser.add_argument("--context-weight", type=float, default=0.0)
    predict_parser.add_argument("--context-short-side", type=int, default=768)
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--manifest", required=True)
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--output", required=True)
    eval_parser.add_argument("--model-dir")
    eval_parser.add_argument("--device", default="cuda")
    eval_parser.add_argument("--tile-size", type=int, default=768)
    eval_parser.add_argument("--stride", type=int, default=512)
    eval_parser.add_argument("--flip", action="store_true")
    eval_parser.add_argument("--limit", type=int, default=0)
    eval_parser.add_argument("--previews", type=int, default=6)
    eval_parser.add_argument("--context-weight", type=float, default=0.0)
    eval_parser.add_argument("--context-short-side", type=int, default=768)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "train":
        config = TeacherConfig(
            **{name: args.pop(name) for name in TeacherConfig.__dataclass_fields__}
        )
        result = train_teacher(
            args.pop("manifest"),
            args.pop("model_dir"),
            args.pop("output"),
            config,
            resume=args.pop("resume"),
        )
    elif command == "predict":
        args["flip"] = not args.pop("no_flip")
        result = predict_teacher(
            args.pop("manifest"), args.pop("checkpoint"), args.pop("output"), **args
        )
    else:
        result = evaluate_checkpoint(
            args.pop("manifest"), args.pop("checkpoint"), args.pop("output"), **args
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
