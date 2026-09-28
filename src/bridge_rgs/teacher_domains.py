"""Audited image-only domain substitution for the fixed teacher split.

Labels, validity, camera grids and view membership remain exactly as in the
original manifest. Rendered masks are never accepted as annotation sources.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MULTI_COMPONENT_DOMAIN = "original_png_multi_component_teacher_domain_v1"
MULTI_COMPONENT_ADAPTER = {
    "id": "original_png_to_legacy_pure_hplus_v1", "border": "constant_zero",
    "interpolation": "INTER_LINEAR", "half_pixel_conjugation": False,
    "rgb_mean": "float32_rint_uint8_0.5",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_image_sources(manifest: dict, *, verify_files: bool = True,
                           verify_validation_files: bool = True) -> dict | None:
    """Validate full manifest lineage before allowing an original-domain warmstart.

    This reads original manifest metadata, TRAIN photographs, and rendered RGB
    only. In particular, it never opens a real validation photograph or any label
    pixels. Annotation path equality is checked against the original manifest.
    """
    protocol = manifest.get("image_source_protocol")
    if protocol is None:
        if any("image_path_sources" in view for view in manifest["views"]):
            raise ValueError("Image sources require an audited source protocol")
        return None
    if protocol.get("id") == MULTI_COMPONENT_DOMAIN:
        return validate_multi_component_sources(manifest, verify_files=verify_files)
    source_path = Path(protocol["original_manifest"])
    if _sha256(source_path) != protocol["original_manifest_sha256"]:
        raise ValueError("Original manifest checksum differs")
    original = json.loads(source_path.read_text())
    if {k: v for k, v in manifest.items() if k not in {"views", "image_source_protocol"}} != {
        k: v for k, v in original.items() if k != "views"
    }:
        raise ValueError("Derived manifest metadata differs from original")
    views = manifest["views"]
    if len(views) != len(original["views"]):
        raise ValueError("Derived manifest view set differs")
    if len({v["name"] for v in views}) != len(views):
        raise ValueError("Derived manifest names must be unique")
    receipts = {}
    for domain in ("train", "val"):
        record = protocol[f"{domain}_render_receipt"]
        if _sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"{domain} render receipt checksum differs")
        receipt = json.loads(Path(record["path"]).read_text())
        if receipt["checkpoint_sha256"] != protocol["renderer_checkpoint_sha256"]:
            raise ValueError("Render receipt checkpoint differs")
        if receipt["source_manifest_sha256"] != protocol["original_manifest_sha256"]:
            raise ValueError("Render receipt original manifest differs")
        if receipt["status"] != "completed" or receipt["grid"] != "pinhole":
            raise ValueError("Render receipt must be complete on the pinhole grid")
        rows = receipt["records"]
        expected = {v["name"] for v in original["views"] if v["split"] == domain}
        if len(rows) != len(expected) or {v["name"] for v in rows} != expected:
            raise ValueError(f"{domain} render receipt view set differs")
        receipts[domain] = {row["name"]: row for row in rows}
    if verify_files and _sha256(protocol["renderer_checkpoint"]) != protocol["renderer_checkpoint_sha256"]:
        raise ValueError("Renderer checkpoint checksum differs")
    image_keys = {"image_path", "image_path_sources", "image_domain"}
    for view, old in zip(views, original["views"], strict=True):
        if {k: v for k, v in view.items() if k not in image_keys} != {
            k: v for k, v in old.items() if k not in image_keys
        }:
            raise ValueError(f"Non-image fields changed: {old['name']}")
        split = view["split"]
        sources = view["image_path_sources"]
        if set(sources) != ({"real", "rendered"} if split == "train" else {"rendered"}):
            raise ValueError("Train needs real/rendered sources; validation is rendered only")
        rendered = receipts[split][view["name"]]
        if sources["rendered"] != {"path": rendered["image_path"], "sha256": rendered["sha256"]}:
            raise ValueError("Rendered RGB source does not match renderer receipt")
        if (rendered["width"], rendered["height"]) != (view["width"], view["height"]):
            raise ValueError("Rendered RGB grid differs from original")
        expected_path = old["image_path"] if split == "train" else rendered["image_path"]
        if view["image_path"] != expected_path:
            raise ValueError("Default image path violates domain policy")
        if split == "train" and sources["real"]["path"] != old["image_path"]:
            raise ValueError("Real train source differs from original")
        if view["image_domain"] != ("mixed_real_rendered" if split == "train" else "rendered_rgb"):
            raise ValueError("Image domain declaration differs")
        if verify_files and (split == "train" or verify_validation_files):
            for source in sources.values():
                if _sha256(source["path"]) != source["sha256"]:
                    raise ValueError(f"Image source checksum differs: {view['name']}")
    return protocol


def _bound_json(record: dict) -> dict:
    if _sha256(record["path"]) != record["sha256"]:
        raise ValueError("Bound domain JSON checksum differs")
    return json.loads(Path(record["path"]).read_text())


def multi_component_receipt(protocol: dict) -> dict:
    """A delivered-image adapter receipt, never a fabricated single renderer."""
    from .coordinates import CORNER, LEGACY, pixel_protocol

    receipt = _bound_json(protocol["cache_receipt"])
    if (protocol["id"] != MULTI_COMPONENT_DOMAIN or receipt.get("id") != MULTI_COMPONENT_DOMAIN
            or receipt.get("status") != "completed"
            or pixel_protocol(protocol) != LEGACY or pixel_protocol(receipt) != LEGACY
            or receipt.get("adapter") != MULTI_COMPONENT_ADAPTER
            or receipt["original_manifest_sha256"] != protocol["original_manifest_sha256"]
            or protocol.get("train_domain") not in {"selected", "composite"}):
        raise ValueError("Invalid explicit multi-component legacy adapter receipt")
    expected = {"selected": LEGACY, "capacity_1m": CORNER, "mcmc": CORNER}
    if set(receipt["components"]) != set(expected):
        raise ValueError("Exactly the three declared source fields are required")
    for key, profile in expected.items():
        if pixel_protocol(receipt["components"][key]) != profile:
            raise ValueError("Component profile cannot be relabeled as the adapter output")
    return receipt


def validate_multi_component_sources(manifest: dict, *, verify_files: bool = True) -> dict:
    """Validate only image substitution on the original legacy mask/camera grid.

    TRAIN real/cache bytes may be checked. VAL paths and advertised SHA are
    bound through the completed cache receipt, but VAL pixels are never opened.
    Both domain variants are in one receipt; validation always uses composite.
    """
    from .coordinates import LEGACY, pixel_protocol

    protocol = manifest["image_source_protocol"]
    receipt = multi_component_receipt(protocol)
    original = _bound_json({"path": protocol["original_manifest"], "sha256": protocol["original_manifest_sha256"]})
    if pixel_protocol(original) != LEGACY or pixel_protocol(manifest) != LEGACY:
        raise ValueError("Multi-component teacher target must retain the original legacy profile")
    if {k: v for k, v in manifest.items() if k not in {"views", "image_source_protocol"}} != {
            k: v for k, v in original.items() if k != "views"}:
        raise ValueError("Derived manifest metadata differs from original")
    if len(manifest["views"]) != len(original["views"]):
        raise ValueError("Derived manifest view set differs")
    names = [v["name"] for v in original["views"]]
    if len(set(names)) != len(names) or any(v["split"] not in {"train", "val"} for v in original["views"]):
        raise ValueError("Duplicate/unknown original split")
    if set(receipt["records"]) != {"train_selected", "train_composite", "val_composite"}:
        raise ValueError("Require both TRAIN domains and composite-only validation")
    caches = {}
    for key, rows in receipt["records"].items():
        split = key.split("_", 1)[0]
        expected = {v["name"]: v for v in original["views"] if v["split"] == split}
        if len(rows) != len(expected) or {r["name"] for r in rows} != set(expected):
            raise ValueError("Cache view population differs from the original split")
        caches[key] = {r["name"]: r for r in rows}
        for row in rows:
            view = expected[row["name"]]
            if (row["width"], row["height"]) != (view["width"], view["height"]):
                raise ValueError("Adapter canvas differs from the unchanged GT grid")
            geometry = row["adapter_info"]
            if (geometry["canvas"] != [view["width"], view["height"]]
                    or not np.array_equal(np.asarray(geometry["canvas_K"], np.float32), np.asarray(view["K"], np.float32))):
                raise ValueError("Adapter intrinsics differ from the unchanged GT grid")
            if verify_files and split == "train" and _sha256(row["image_path"]) != row["sha256"]:
                raise ValueError("TRAIN cache RGB checksum differs")
    image_keys = {"image_path", "image_path_sources", "image_domain"}
    for view, old in zip(manifest["views"], original["views"], strict=True):
        if {k: v for k, v in view.items() if k not in image_keys} != {
                k: v for k, v in old.items() if k not in image_keys}:
            raise ValueError(f"Non-image fields changed: {old['name']}")
        train = old["split"] == "train"
        cache = caches[f"train_{protocol['train_domain']}" if train else "val_composite"][old["name"]]
        sources = view["image_path_sources"]
        if set(sources) != ({"real", "rendered"} if train else {"rendered"}):
            raise ValueError("Train needs real/rendered; validation composite only")
        if sources["rendered"] != {"path": cache["image_path"], "sha256": cache["sha256"]}:
            raise ValueError("Rendered source differs from explicit adapter cache")
        if view["image_domain"] != ("mixed_real_rendered" if train else "rendered_rgb"):
            raise ValueError("Image domain differs")
        if view["image_path"] != (old["image_path"] if train else cache["image_path"]):
            raise ValueError("Default image source differs")
        if train:
            if sources["real"]["path"] != old["image_path"]:
                raise ValueError("Real TRAIN source differs")
            if verify_files and _sha256(old["image_path"]) != sources["real"]["sha256"]:
                raise ValueError("Real TRAIN RGB checksum differs")
    return protocol


def verify_multi_component_renderers(protocol: dict) -> None:
    """Read actual component metadata in its own profile, not the output profile."""
    import torch

    from .coordinates import CORNER, pixel_protocol

    receipt = multi_component_receipt(protocol)
    for component in receipt["components"].values():
        if _sha256(component["checkpoint"]) != component["checkpoint_sha256"]:
            raise ValueError("Component checkpoint checksum differs")
        state = torch.load(component["checkpoint"], map_location="cpu", weights_only=False)
        if pixel_protocol(state) != pixel_protocol(component):
            raise ValueError("Actual component checkpoint profile differs")
        declaration = state.get("manifest_sha256")
        observed = component.get("source_manifest_sha256")
        if declaration is not None and observed != declaration:
            raise ValueError("Component own manifest declaration differs")
        if pixel_protocol(component) == CORNER and (not declaration or not observed):
            raise ValueError("Corner component requires its own manifest SHA")


def verify_common_original_warmstart(initial: dict, protocol: dict) -> dict:
    """Allow an audited old derived teacher domain to share the same original.

    This exception is only for the explicit adapter schema. It validates the
    original and the historical derived manifest before exposing EMA weights;
    it neither rewrites their profile nor weakens ordinary warmstart/resume.
    """
    from .coordinates import LEGACY, pixel_protocol

    if protocol.get("id") != MULTI_COMPONENT_DOMAIN:
        raise ValueError("Common-original warmstart requires explicit adapter schema")
    provenance = initial["provenance"]
    historical = _bound_json({"path": provenance["manifest"], "sha256": provenance["manifest_sha256"]})
    old_protocol = historical.get("image_source_protocol")
    if old_protocol != provenance.get("image_source_protocol") or old_protocol is None:
        raise ValueError("Historical teacher domain provenance differs")
    if (old_protocol["original_manifest_sha256"] != protocol["original_manifest_sha256"]
            or pixel_protocol(historical) != LEGACY or pixel_protocol(initial) != LEGACY
            or pixel_protocol(provenance) != LEGACY):
        raise ValueError("Warmstart common original/profile differs")
    # Historical receipt/SHA validation reads rendered VAL RGB only in its old
    # validator. Skip pixel validation here: no validation pixel payload is needed.
    validate_image_sources(historical, verify_files=False)
    return {"policy": "verified historical derived domain -> same original legacy; EMA only",
            "historical_manifest": provenance["manifest"], "historical_manifest_sha256": provenance["manifest_sha256"],
            "common_original_manifest_sha256": protocol["original_manifest_sha256"]}


def domain_schedule(steps: int, rendered_probability: float, seed: int) -> np.ndarray:
    """A predeclared exact-count schedule, independent of crop/view RNG."""
    if steps < 1 or not 0 <= rendered_probability <= 1:
        raise ValueError("Invalid image domain schedule")
    schedule = np.zeros(steps, dtype=np.uint8)
    schedule[:round(steps * rendered_probability)] = 1
    np.random.default_rng(seed + 77341).shuffle(schedule)
    return schedule


def select_image_source(view: dict, rendered: bool) -> dict:
    if view["split"] != "train":
        raise ValueError("Domain mixing is only allowed for train views")
    domain = "rendered" if rendered else "real"
    source = view["image_path_sources"][domain]
    return {**view, "image_path": source["path"], "image_domain": domain}
