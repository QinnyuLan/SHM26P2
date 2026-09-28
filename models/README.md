# DINOv3 teacher weights

The current semantic teacher is ModelScope
`facebook/dinov3-vith16plus-pretrain-lvd1689m` (DINOv3 ViT-H+/16). The weight is
stored in the [SHM2026 ModelScope dataset](https://modelscope.cn/datasets/sky931/SHM2026/files),
not in Git. Download or refresh all release assets with:

```bash
uv run python scripts/download_dinov3.py --output models/dinov3-vith16plus
```

The expected `model.safetensors` SHA-256 is recorded in the ModelScope release
manifest and deployment bundle. Do not replace it without updating the hash
and model provenance.
