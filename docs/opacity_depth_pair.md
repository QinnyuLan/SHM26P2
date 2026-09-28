# Opacity-only 稀疏深度配对：1,000 步

从 `runs/strong_semantic_coupled/last.pt`（语义 8k、498,136 Gaussians）开始，两组均只更新 opacity，保持位置、形状、RGB SH、背景色、Gaussian semantic features、decoder、refiner、prior_counts 与全部相机不动。两组都使用完整 350 TRAIN 的独立 shuffle RNG seed42、原生1320×989、完整SH3、相同RGB目标；只有 `sparse_depth_weight` 分别为 0 与 0.05。没有重置 head，没有语义训练、教师监督、融合或增密。权重0组不构建稀疏深度分支。

第一组由 `scripts/run_experiment.py` 固化源码，第二组显式复用其 `source_snapshot`。CPU核对两组源码/common input SHA一致，仅 output 和深度权重不同；最终 sampler order/cursor/RNG 完全相同。与起点比较，唯一改变的模型 tensor 均为 `splats.opacity_logits`，相机及所有其他 tensor 逐位相同。depth组另记录原始tracks SHA。审计：`runs/opacity_pair_audit.json`。

| 原生验证指标 | 8k 起点 | 1k RGB control | 1k RGB + SfM depth |
|---|---:|---:|---:|
| PSNR ↑ | 30.27351 | 30.34757 | 30.34783 |
| SSIM ↑ | 0.902818 | 0.903026 | 0.902983 |
| LPIPS ↓ | 0.224494 | 0.224156 | 0.224062 |
| final all5 mIoU (%) | 93.90821 | 93.87992 | 93.90556 |
| final cable IoU (%) | 93.64356 | 93.49856 | 93.46280 |
| final tower IoU (%) | 91.31644 | 91.50089 | 91.66974 |
| final foundation IoU (%) | 89.06267 | 88.89969 | 88.89616 |
| raw all5 mIoU (%) | 78.64341 | 78.69715 | 78.81815 |
| raw cable IoU (%) | 29.78754 | 30.29640 | 30.56312 |
| 235 PSNR | 20.45534 | 20.18629 | 20.19846 |
| 235 final all5 mIoU (%) | 56.77167 | 56.09908 | 56.85679 |

三组 evaluation fingerprint 一致，全部 50 RGB/41有GT验证视角保留。与 matched RGB control 相比，稀疏深度项带来 final all5 **+0.02563 pp**、cable **−0.03576 pp**、raw all5 **+0.1210 pp**；没有证明实用的总体语义提升，也没有修复 235 的大面积 RGB 浮层。与8k起点比较，depth组 final all5 基本持平。这里没有多seed/置信区间，不把微小差值解释为稳定优势。

初次真实 CUDA 两步预检确认深度项本身有非零、有限的 opacity 梯度；正式训练末步有效目标1,139，median relative depth error为0.04386，过近比例0.08867。这是当步训练视角诊断，不能与另一个视角/步骤的误差直接比较作为收敛证据。完整几何门槛、稀疏支持覆盖、原始UV恢复策略与经典先例见 `docs/ray_support.md`；本项仍是经典SfM深度控制，不是创新。

语义参数没有训练；raw/final输出的变化来自 opacity 改变后渲染混合及 refiner 输入变化，不能声称 Gaussian 点语义类别本身被改进。Expected-depth 的前后层抵消限制仍在；本实验不支持直接把该步骤并入默认最终管线。后续若研究射线分布或增密数学修复，应独立设计控制，不把新机制的收益回记到这组深度实验。

产物：`runs/opacity_rgb_control/last.pt`、`runs/opacity_sfm_depth/last.pt`；原生完整评测均在各自 `evaluation_native/metrics.json`。两组GPU进程已完成并退出。
