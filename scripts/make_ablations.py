"""Generate matched-data experiments; these configs do not imply measured gains."""

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="configs/bridge_rgs.yaml")
    parser.add_argument("--output", default="configs/ablations")
    args = parser.parse_args()
    base = yaml.safe_load(Path(args.base).read_text())
    variants = {
        "01_fixed_camera_supervised": {
            "pseudo_dir": None,
            "multiview_fusion": False,
            "semantic_geometry_weight": 0,
            "densification": "absgrad",
            "structure_guided": False,
            "region_rgb_weight": 0,
        },
        "02_plain_pseudo": {
            "multiview_fusion": False,
            "semantic_geometry_weight": 0,
            "densification": "absgrad",
            "structure_guided": False,
        },
        "03_multiview_point": {
            "projection_uncertainty": False,
            "semantic_geometry_weight": 0,
            "densification": "absgrad",
            "structure_guided": False,
        },
        "04_projection_uncertainty": {
            "semantic_geometry_weight": 0,
            "densification": "absgrad",
            "structure_guided": False,
        },
        "05_compensated_residual": {"semantic_geometry_weight": 0, "structure_guided": False},
        "06_full_method": {},
        "07_full_camera_updates": {"optimize_cameras": True},
        "08_full_no_refiner": {"refine_start": base["steps"] + 1},
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, overrides in variants.items():
        config = {**base, **overrides, "output": f"runs/ablations/{name}"}
        (output / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Wrote {len(variants)} configs to {output}; same split, teacher and max Gaussians")


if __name__ == "__main__":
    main()
