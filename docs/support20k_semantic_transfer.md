# 同训练链 30k → 20k 几何的固定 1NN 语义迁移

本次零训练工程控制保留了大部分源语义能力，但没有超过源模型：全 5 类 mIoU 从 94.5118% 降至 **93.8566%**，索从 93.5493% 降至 **92.1629%**。不能作为新的性能收益或学术创新。它比此前不同几何链的 69k→498k 迁移更可用，但两次实验的输入与方向不同，不能把差距归因于某个新算法。

## 固定协议与来源

- 源：`runs/support_split_semantic_coupled/last.pt`，498136 个高斯，同链完整 RGB 30k 后冻结几何训练 8k 语义；SHA256 `a329fc70d582972fc338715765304b99da208c821ee34288809db500d5acd6b9`。
- 目标：`runs/support20k_appearance_polish/last.pt`，425447 个高斯，同链 RGB 20k 后仅颜色优化 3k；SHA256 `88f1c0577bc2bd02a23a5255151b4f0d45545c2f813fccef4b673a2465757c73`。
- 两个输入均为已完成 run 的最终文件，哈希与原 receipt 匹配；manifest SHA、350 TRAIN 相机、scene_scale=16.261470794677734、feature_dim=16 完全一致。
- 原样复用 `runs/semantic_transfer_1nn/source_snapshot` 的 21 个 Python 文件，全部 SHA 一致；没有改动主训练代码。固定 Euclidean 1NN，精确等距时取源索引最小值。没有 k 搜索、距离接受阈值、空间平滑或训练。
- 仅迁移 `splats.sem_features`、完整 `semantic_decoder.*`、完整 `refiner.*` 及对应多尺度头结构元数据。目标 means/quats/scales/opacity/SH/背景、`semantic_prior_counts` 和训练相机均逐位保留；源 decoder/refiner 逐位复制，映射 feature 张量再次按固定 index 验证。
- 派生输出 `runs/support20k_semantic_transfer/last.pt` 明确为 inference/warmstart-only，step=0，无优化器状态，不能当作原训练的严格 resume。

目标的原 20k 输入 checkpoint 已在先前清理中缺失，但本次直接输入是保留完好的 appearance20k `last.pt`，没有重建或冒用已缺失文件。其来源保留限制见 [floaters_and_split_audit.md](floaters_and_split_audit.md)。

## 映射诊断

| 量 | 本次固定 1NN |
|---|---:|
| 最近邻世界距离 p50 / p90 / p95 / p99 | 0.004175 / 0.016515 / 0.026807 / 0.120894 |
| 距离除以 scene_scale，p50 / p99 | 0.000257 / 0.007434 |
| 最大世界距离 | 7.201299 |
| 精确零距离目标数 | 18296 |
| 实际使用的不同源点 | 356618（源点的71.59%） |
| 超过首次映射的重复指派数 | 68829 |
| 与其他目标共享同一源点的目标数 | 125549（29.51%） |
| 单一源点最大目标指派数 | 512 |
| 精确等距、按索引确定性消歧数 | 3074 |
| 位于最近源点 3σ 形状椭球内 | 58.39% |

距离单位是 COLMAP 任意世界尺度，不是米。此前 69k→498k 控制的 p50/p99 距离为 0.03705/3.97307，本次明显更近；但 3σ 使用源高斯形状而非位置协方差，不能解释为语义正确概率，也没有用其筛点。更近的中心距离并不保证完全一致的遮挡或渲染特征分布。

`mapping_diagnostics.npz` 保存所有目标的源 index、世界距离、所有源的指派次数；`mapping_summary.json` 保存完整碰撞直方图与张量不变性。没有按验证标签选择映射策略。

## 完整原生评价

固定 original-camera、1320×989、全部 50 个 RGB 视角及 41 个语义有效视角，含 LPIPS。评价指纹与源/目标相同：`750b9f53046bc104093715c6c26c090837746c467445364584570feb21f10b7b`。

| 指标 | 30k 语义源 | 20k 外观目标 + 1NN，零训练 |
|---|---:|---:|
| 高斯数 | 498136 | 425447 |
| PSNR ↑ | 30.102410 | 29.943618 |
| SSIM ↑ | 0.902981 | 0.896792 |
| LPIPS ↓ | 0.224007 | 0.238845 |
| final 全 5 类 mIoU（%） | 94.5118 | 93.8566 |
| final 前景 mIoU（%） | 93.3197 | 92.5380 |
| background IoU（%） | 99.2799 | 99.1309 |
| deck IoU（%） | 96.9947 | 96.6276 |
| cable IoU（%） | 93.5493 | 92.1629 |
| tower IoU（%） | 93.9470 | 92.8407 |
| foundation IoU（%） | 88.7879 | 88.5210 |
| raw 3D 全 5 类 mIoU（%） | 79.1987 | 78.8897 |
| raw 3D cable IoU（%） | 30.8081 | 30.3714 |

迁移后全部 50 张 RGB PNG 与 appearance20k 目标逐字节相同，PSNR/SSIM/LPIPS 也完全一致。因此 RGB 差异来自已存在的目标几何/外观，本次没有改变 RGB 或修复浮层。目标未经语义训练的随机头成绩不作为有意义的训练基线。

配对视角 bootstrap 5000 次、种子 20260926，以迁移减语义源计算：

| 指标 | 差值（百分点） | 95% 配对视角区间（百分点） |
|---|---:|---:|
| final 全 5 类 mIoU | -0.6551 | [-1.2925, +0.3120] |
| final 前景 mIoU | -0.7817 | [-1.5599, +0.4073] |
| cable IoU | -1.3864 | [-2.4738, -0.4215] |
| tower IoU | -1.1063 | [-1.6253, -0.4793] |
| raw 3D 全 5 类 mIoU | -0.3089 | [-0.9267, +0.6123] |

整体均值区间跨零，但索和塔身的条件区间均为负。不能把“保存了多数能力”解释为完全等价，更不能仅挑选一个受益视角。这里是单场景开发视角的条件重采样，训练预算、点数、几何和外观都不同，不构成同预算算法比较或跨场景显著性。

## 留存与复现

`evaluation_receipt.json` 记录 detached 评估命令和返回码；`evaluation_summary.json` 验证输入文件未变化、最终 checkpoint/metrics SHA、50张 RGB 不变性与评价视角数量。派生模型 SHA256 为 `885a7f89ead1099173a7fe7ba3fdda7c66825510935b7ed99b65f0d628d9cfe9`。本次目录总占用约 192 MB；所有输入 last、输出、source、日志、映射诊断均保留，没有清理任何文件。

```bash
PYTHONPATH=runs/semantic_transfer_1nn/source_snapshot uv run python scripts/transfer_semantic_1nn.py \
  --source runs/support_split_semantic_coupled/last.pt \
  --target runs/support20k_appearance_polish/last.pt \
  --output runs/support20k_semantic_transfer/last.pt
PYTHONPATH=runs/support20k_semantic_transfer/source_snapshot uv run python -m bridge_rgs.cli evaluate \
  --checkpoint runs/support20k_semantic_transfer/last.pt \
  --manifest artifacts/prepared/manifest.json \
  --output runs/support20k_semantic_transfer/evaluation_native --scale 1 --lpips
```

迁移脚本拒绝覆盖现存输出；独立复现应选择新的输出目录。没有启动后续训练，也没有根据本次验证结果调 k、阈值或语义头。
