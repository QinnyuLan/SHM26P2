"""Make a controlled training-pose perturbation; validation cameras/pixels stay fixed."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from bridge_rgs.reliability import se3_exp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=Path("configs/strong_rgb_sanity.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pose_stress_mild"))
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--translation-sigma-fraction", type=float, default=.0005)
    parser.add_argument("--rotation-sigma-radians", type=float, default=.001)
    args = parser.parse_args()
    if args.translation_sigma_fraction < 0 or args.rotation_sigma_radians < 0:
        raise ValueError("Pose perturbation scales must be nonnegative")
    config = yaml.safe_load(args.base_config.read_text())
    source = Path(config["manifest"])
    original = json.loads(source.read_text())
    manifest = copy.deepcopy(original)
    sigma_translation = args.translation_sigma_fraction * float(manifest["scene_radius"])
    rng = np.random.default_rng(args.seed)
    anchored = False
    injected = {}
    for view in manifest["views"]:
        if view["split"] != "train":
            continue
        if not anchored:
            anchored = True
            continue
        delta = rng.normal(size=6) * np.array([sigma_translation]*3 + [args.rotation_sigma_radians]*3)
        perturbation = se3_exp(torch.tensor(delta, dtype=torch.float64)).numpy()
        view["w2c"] = (perturbation @ np.array(view["w2c"])).tolist()
        injected[view["name"]] = delta.tolist()
    assert [v for v in manifest["views"] if v["split"] == "val"] == [v for v in original["views"] if v["split"] == "val"]
    manifest["pose_stress"] = {"source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                               "seed": args.seed, "translation_sigma_world": sigma_translation,
                               "rotation_sigma_radians": args.rotation_sigma_radians,
                               "injected_left_twists_translation_first": injected,
                               "scope": "training extrinsics only; first train camera anchored; clean initialization held fixed"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to replace stress manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    config.update(manifest=str(manifest_path),
                  init_points=config.get("init_points", str(source.with_name("init_points.npz"))),
                  pseudo_dir=None, multiview_fusion=False, optimize_cameras=False)
    for use_attribution, label in ((True, "compensated"), (False, "raw_residual")):
        candidate = dict(config, camera_attribution=use_attribution,
                         output=f"runs/{args.output_dir.name}_{label}")
        path = args.output_dir / f"{label}.yaml"
        path.write_text(yaml.safe_dump(candidate, sort_keys=False))
        print(path)


if __name__ == "__main__":
    main()
