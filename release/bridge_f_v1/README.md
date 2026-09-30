# Bridge-RGS multi-field release

This directory records the reproducible model contract for the selected multi-field
deployment. The corresponding dataset, DINOv3 weights, and checkpoints are
published at [ModelScope `sky931/SHM2026`](https://modelscope.cn/datasets/sky931/SHM2026/files).
The source repository keeps only this lightweight contract; render caches and
historical experiment checkpoints are excluded.

## Components

| Component | Role | SHA-256 |
|---|---|---|
| `h3_moments/02_cross/last.pt` | semantic depth-moment field | `22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226` |
| `rgb_mcmc_reference_500k/last.pt` | RGB MCMC field | `3796c4c73ce2189ce74153dd7b1209d88cc2d9a1bf57c9299366977cde8c876b` |
| `teacher_render_adapt_v1/best.pt` | DINOv3 semantic decoder | `00f5b84ac9a56c39512c5b8e43f70110397923feea1a2b4c78bdffd1f5524bff` |
| `dinov3-vith16plus/model.safetensors` | frozen DINOv3 ViT-H+/16 | `3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d` |
| `ibgs_layer_heads_matched_v1/top4_normalized/last.pt` | normalized top-4 RGB head | `4b13a8aa7501b6cb42e5bdf40d8ea9334e5884f83cc13ff12ff59f7c3fa46346` |

The exact local source paths used for the archived evaluation are preserved in
the historical bundle outside Git. Download the assets with
`scripts/download_modelscope_assets.py`; never edit the recorded hashes.

## Reported reference

On the fixed 50/41 held-out protocol, F reports PSNR 30.236593 dB, SSIM
0.875024, LPIPS 0.259300, and five-class mIoU 95.109016%. These are supervised
held-out results under the documented sensor and split contract, not a claim
of generalization to a different dataset.
