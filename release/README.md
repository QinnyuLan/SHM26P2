# Bridge-RGS submission package

This directory contains the six engineering materials required for the blind
camera-rendering submission. Every command below is run from the repository
root and uses the project `uv` environment.

## Contents

1. `code/` contains the release wrappers and a frozen copy of the Python
   renderer/trainer package. The wrappers include English comments and perform
   input and output checks.
2. `README.md` documents installation, data preparation, training, rendering,
   coordinate conventions, output files, and failure conditions.
3. `data/README.md` documents the public dataset and its checksum procedure.
4. `checkpoints/` contains the checkpoint manifest. The binary checkpoints are
   downloaded from ModelScope and are intentionally excluded from Git.
5. `code/render_blind.py` is the blind rendering entry point. It reads camera
   parameters only and writes an RGB PNG and a semantic-ID PNG for every view.
6. `configs/` contains the exact full-view training recipes and the inference
   contract. `VERIFICATION.md` records the pre-submission checks.

The presentation, paper, and demonstration videos are maintained separately in
`report/`. This package is the executable Project 2 submission core. The
archived multi-field contract is documented under `bridge_f_v1/`.

## Environment

```bash
uv sync --python 3.11
```

The tested environment uses PyTorch 2.8.0 + CUDA 12.8, gsplat 1.5.3, and a
CUDA-capable NVIDIA GPU. CPU-only machines can inspect the files and run the
schema checks, but cannot train or render the Gaussian field.

## Download the public assets

The data and checkpoints are hosted in the public
[ModelScope dataset `sky931/SHM2026`](https://modelscope.cn/datasets/sky931/SHM2026/files).
The downloader also verifies every file listed in `manifest.sha256`.

```bash
uv run python scripts/download_modelscope_assets.py \
  --output release_assets/SHM2026
```

The full-view checkpoints are under
`release_assets/SHM2026/checkpoints/bridge_full400_v1/`:

| File | Purpose |
|---|---|
| `rgb_full400.pt` | 30,000-step RGB field, 498,136 Gaussians |
| `semantic_full400_base.pt` | 8,000-step semantic warm-start |
| `semantic_full400_cross.pt` | Final 3,000-step depth-feature cross-moment semantic field |

The checkpoint manifest records the SHA-256 values and the expected ModelScope
paths. The final semantic checkpoint is the default blind-rendering model. The
full-view blind path directly renders this trained semantic Gaussian field; it
does not run a student-teacher distillation step. Where the archived
multi-field bundle uses DINOv3, DINOv3 is a frozen pretrained visual encoder
with a semantic decoder, not a teacher used to train a student.

## Blind rendering

The renderer accepts a JSON file containing camera parameters only. It never
opens target RGB images, masks, labels, or dataset annotations.

```bash
uv run python release/code/render_blind.py \
  --checkpoint release_assets/SHM2026/checkpoints/bridge_full400_v1/semantic_full400_cross.pt \
  --cameras path/to/test_cameras.json \
  --output submission_outputs \
  --grid official
```

The input may be a list of camera records or an object with a `views` list.
Each record must contain:

```json
{
  "name": "test_001.png",
  "K": [[925.7016, 0.0, 660.0], [0.0, 925.7016, 494.5], [0.0, 0.0, 1.0]],
  "w2c": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
  "width": 1320,
  "height": 989,
  "distortion": [0.008987863345268233, 0.0, 0.0, 0.0, 0.0]
}
```

`w2c` is the COLMAP/OpenCV world-to-camera matrix. The official sensor is
1320×989 with the supplied calibrated intrinsics and distortion. The renderer
uses the checkpoint's `colmap_corner_v2` pixel convention, creates an overscan
pinhole canvas for distorted cameras, and warps the result back to the official
grid. `--grid pinhole` ignores distortion and is intended only for a camera
file that already describes an undistorted pinhole image.

The output contains `rgb/<name>.png`, `mask/<name>.png`, and
`render_receipt.json`. RGB is uint8 sRGB. The mask is a single-channel uint8
image with exactly these IDs: 0 background, 1 deck, 2 stay cable, 3 tower,
and 4 foundation. Unknown IDs are never emitted.

## Reproduce the full-view training

The training wrapper creates a new all-view preparation directory and runs the
RGB, semantic, and cross-moment stages in order. It does not use validation
images because all 400 views are deliberately fitted.

```bash
uv run python release/code/train_full400.py \
  --dataset release_assets/SHM2026/dataset \
  --artifact-root artifacts/release_full400 \
  --run-root runs/release_full400
```

Use `--dry-run` to inspect the commands without reading data or starting a GPU
job. Use a fresh run directory for a clean reproduction. `--skip-prepare` is available
when an existing preparation directory and completed warm-start stages are
intentionally reused.

## Verification before submission

```bash
uv run python release/code/render_blind.py --help
uv run python -m compileall -q release/code
uv run ruff check release/code
uv run python release/code/verify_release.py --assets-root release_assets/SHM2026
uv run pytest -q tests/test_evaluation_protocol.py tests/test_official_evaluate.py
```

The renderer performs additional checks at runtime: unique safe filenames,
finite 3×3 intrinsics, finite homogeneous poses, positive dimensions,
checkpoint loading, output dimensions, and the mask ID range. A failed check
stops the run before producing a misleading submission receipt.
