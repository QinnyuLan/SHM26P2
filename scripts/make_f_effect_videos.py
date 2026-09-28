"""Encode RGB and colorized semantic-mask videos from a completed F render."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

# background, deck, stay cable, tower, foundation
PALETTE = np.array(
    [[35, 42, 52], [230, 160, 35], [225, 55, 65], [70, 130, 225], [110, 200, 125]],
    dtype=np.uint8,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(pattern: Path, output: Path, fps: int) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(pattern),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:black",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-root",
        type=Path,
        default=Path("/mnt/data/SHM2026/runs/f_all300_evaluation_v1/render"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data/SHM2026/runs/f_all300_evaluation_v1/videos"),
    )
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    if args.fps < 1:
        raise ValueError("fps must be positive")
    render_root = args.render_root.resolve()
    output_dir = args.output_dir.resolve()
    rgb_dir, mask_dir = render_root / "rgb", render_root / "mask"
    rgb_names = sorted(rgb_dir.glob("*.png"))
    mask_names = sorted(mask_dir.glob("*.png"))
    expected = [f"{index:03d}.png" for index in range(1, 301)]
    if [path.name for path in rgb_names] != expected or [path.name for path in mask_names] != expected:
        raise ValueError("Expected contiguous 001.png..300.png RGB and mask frames")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    color_dir = output_dir / "mask_color"
    color_dir.mkdir(parents=True)
    for path in mask_names:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (989, 1320) or int(mask.max()) > 4:
            raise ValueError(f"Invalid semantic frame: {path}")
        color = PALETTE[mask]
        cv2.imwrite(str(color_dir / path.name), color[..., ::-1])

    rgb_video = output_dir / "bridge_rgb_f.mp4"
    mask_video = output_dir / "bridge_semantic_mask_f.mp4"
    encode(rgb_dir / "%03d.png", rgb_video, args.fps)
    encode(color_dir / "%03d.png", mask_video, args.fps)

    result = {
        "status": "completed",
        "render_root": str(render_root),
        "frame_count": 300,
        "fps": args.fps,
        "duration_seconds": 300 / args.fps,
        "source_resolution": [1320, 989],
        "video_resolution": [1320, 990],
        "ordering": "numeric camera names 001.png..300.png",
        "palette_rgb": PALETTE.tolist(),
        "rgb_video": {"path": str(rgb_video), "sha256": sha(rgb_video)},
        "semantic_video": {"path": str(mask_video), "sha256": sha(mask_video)},
    }
    (output_dir / "video_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
