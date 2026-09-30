# Dataset material

The released dataset is the 400-view bridge scene in the public ModelScope
repository:

<https://modelscope.cn/datasets/sky931/SHM2026/files>

Download it with:

```bash
uv run python scripts/download_modelscope_assets.py \
  --output release_assets/SHM2026
```

The downloaded tree contains `dataset/images/`,
`dataset/unlabeled_Images/`, `dataset/json/`, and
`dataset/camera_parameters/`. The camera parameters are the official COLMAP
reference. The release downloader verifies the complete tree against the
repository `manifest.sha256`; no private local path is required.

The training split used for the full-view checkpoint contains all 400 views
(300 annotated and 100 RGB-only). The paper's independent protocol remains the
fixed 350/50 split and is not redrawn by this release package.
