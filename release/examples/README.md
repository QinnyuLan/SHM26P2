# Example blind-test camera file

`cameras.json` is a target-only camera file containing two official 1320 x 989 COLMAP views. It contains no target RGB image, mask, or annotation path. Replace these records with the hidden test camera parameters supplied by the evaluation service.

Each record supplies a unique output name, a 3 x 3 intrinsic matrix `K`, a world-to-camera pose `w2c`, image dimensions, and the five OpenCV distortion coefficients `[k1, k2, p1, p2, k3]`. The pose uses the official COLMAP/OpenCV convention documented in `../configs/render_contract.json`.

Example invocation from the repository root:

```bash
uv run python release/code/render_blind.py \
  --checkpoint release_assets/SHM2026/checkpoints/bridge_full400_v1/semantic_full400_cross.pt \
  --cameras release/examples/cameras.json \
  --output /tmp/bridge_rgs_render \
  --grid official
```

The output contains `rgb/*.png`, `mask/*.png`, and a machine-readable `render_receipt.json`.
