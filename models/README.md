# DINOv3 teacher weights

The current semantic teacher is ModelScope
`facebook/dinov3-vith16plus-pretrain-lvd1689m` (DINOv3 ViT-H+/16). The selected
weight is intentionally included in this repository as a Git LFS release
asset. Download or refresh it with the project helper:

```bash
uv run python scripts/download_dinov3.py --output models/dinov3-vith16plus
```

The expected `model.safetensors` SHA-256 is recorded in the release manifest
and deployment bundle. Do not replace it without updating the hash and model
provenance.
