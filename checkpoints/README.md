# Trained checkpoints

Training checkpoints are release assets rather than Git source files. They are
large, machine-specific binary files and are ignored by `.gitignore`. A
public release should publish them through an artifact store or GitHub Release
and place the downloaded files under this directory.

The selected F deployment uses the component hashes and paths documented in
[`release/bridge_f_v1/README.md`](../release/bridge_f_v1/README.md). Verify
SHA-256 before using a checkpoint.

## 全量 400 视角版本

ModelScope 数据集 `sky931/SHM2026` 的
`checkpoints/bridge_full400_v1/` 保存全量 400 视角训练权重：

- `rgb_full400.pt`：30,000 步 RGB 场，498,136 个 Gaussian；
- `semantic_full400_base.pt`：8,000 步语义 warm-start；
- `semantic_full400_cross.pt`：最终 3,000 步深度—特征交叉矩精修端点。

全量模型使用所有 400 个视角训练，相关 50/41 结果属于训练重叠诊断，不能替代固定 350/50 留出评价。
