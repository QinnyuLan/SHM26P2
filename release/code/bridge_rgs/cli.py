"""Bridge-RGS reproducible command-line entry point."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Audit, undistort and triangulate training-only tracks")
    prepare.add_argument("--dataset", default="Dataset")
    prepare.add_argument("--output", default="artifacts/prepared")
    prepare.add_argument("--max-width", type=int, default=1320)
    prepare.add_argument("--max-points", type=int, default=60000)
    prepare.add_argument("--val-every", type=int, default=8)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--workers", type=int, default=8)
    prepare.add_argument("--ba", action="store_true")
    prepare.add_argument("--pixel-protocol", choices=["legacy_mixed_v1", "colmap_corner_v2"],
                         default="colmap_corner_v2",
                         help="New data uses corrected pixel centers; legacy is for historical reproduction")
    train = sub.add_parser("train", help="Train semantic Gaussian reconstruction")
    train.add_argument("--config", default="configs/bridge_rgs.yaml")
    train.add_argument("--resume")
    train.add_argument("--steps", type=int)
    train.add_argument("--output")
    evaluate = sub.add_parser("evaluate", help="Score original held-out cameras")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--manifest", default="artifacts/prepared/manifest.json")
    evaluate.add_argument("--output", default="runs/evaluation")
    evaluate.add_argument("--max-views", type=int)
    evaluate.add_argument("--scale", type=float, default=1.)
    evaluate.add_argument("--lpips", action="store_true")
    evaluate.add_argument("--refiner-flip-tta", action="store_true")
    render = sub.add_parser("render", help="Render camera parameters only into official ID PNGs")
    render.add_argument("--checkpoint", required=True)
    render.add_argument("--cameras", required=True)
    render.add_argument("--output", default="runs/submission")
    render.add_argument("--grid", choices=["official", "pinhole"], default="official")
    render.add_argument("--max-views", type=int)
    render.add_argument("--teacher-checkpoint",
                        help="Optional costly fixed 50/50 DINOv3 combination on rendered RGB")
    render.add_argument("--refiner-flip-tta", action="store_true",
                        help="Optional fixed two-pass horizontal refiner augmentation")
    export = sub.add_parser("export-ply", help="Export explicit semantic 3D Gaussians")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", default="runs/bridge_semantic.ply")
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare_dataset
        result = prepare_dataset(args.dataset, args.output, max_width=args.max_width,
                                 max_points=args.max_points, val_every=args.val_every,
                                 bundle_adjustment=args.ba, seed=args.seed, workers=args.workers,
                                 pixel_protocol=args.pixel_protocol)
    elif args.command == "train":
        from .io import read_config
        from .train import train as run
        config = read_config(args.config)
        if args.steps:
            config["steps"] = args.steps
        if args.output:
            config["output"] = args.output
        result = run(config, args.resume)
    elif args.command == "evaluate":
        from .evaluate import evaluate
        result = evaluate(args.checkpoint, args.manifest, args.output, max_views=args.max_views,
                          scale=args.scale, lpips_metric=args.lpips, refiner_flip_tta=args.refiner_flip_tta)
    elif args.command == "render":
        from .evaluate import render_cameras
        result = render_cameras(args.checkpoint, args.cameras, args.output, args.grid, args.max_views,
                                teacher_checkpoint=args.teacher_checkpoint, refiner_flip_tta=args.refiner_flip_tta)
    else:
        from .export import export_ply
        result = export_ply(args.checkpoint, args.output)
    print(str(result) if isinstance(result, Path) else json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
