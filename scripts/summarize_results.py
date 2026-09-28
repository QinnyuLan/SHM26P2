"""Collect measured evaluation receipts without inventing a competition score."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="metrics.json files or folders")
    parser.add_argument("--output", default="runs/results.csv")
    args = parser.parse_args()
    records = []
    for raw in args.paths:
        path = Path(raw)
        files = sorted(path.rglob("metrics.json")) if path.is_dir() else [path]
        for file in files:
            metrics = json.loads(file.read_text())
            row = {
                "source": str(file),
                **{
                    key: metrics.get(key)
                    for key in [
                        "validation_views",
                        "gaussians",
                        "psnr",
                        "ssim",
                        "lpips",
                        "miou_foreground",
                        "miou_all",
                        "mean_render_seconds",
                    ]
                },
            }
            records.append(row)
    if not records:
        raise SystemExit("No actual metric receipts found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(output)


if __name__ == "__main__":
    main()
