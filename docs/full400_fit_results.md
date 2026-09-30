# 全量 400 视角训练与评估记录

本记录对应一次独立的全量拟合实验。400 个官方相机视角全部用于训练，因此它用于生成最终模型和测量训练拟合质量；它不能替代论文中的 350/50 固定划分，也不能作为独立泛化指标。

## 训练配置

- 数据准备：`artifacts/prepared_corner_v2_full400/manifest.json`，`val_every=0`，corner-v2 像素协议，60,000 个三角化初始化点。
- RGB 场：30,000 步，混合增密，最大 500,000 个 Gaussian；最终 498,136 个。
- 语义场：从全量 RGB 场 warm-start，8,000 步，仅使用 300 张带标注图的监督信号。
- 深度—特征交叉矩精修：冻结几何、RGB 和相机，仅更新精修器 3,000 步。
- 未启用伪标签、多视图软融合和语义几何损失；无标注图像通过 RGB 训练参与外观建模。

配置文件为 [full400_rgb.yaml](../configs/full400_rgb.yaml)、[full400_semantic.yaml](../configs/full400_semantic.yaml) 和 [full400_semantic_moments_cross.yaml](../configs/full400_semantic_moments_cross.yaml)。权重和逐步日志位于被 Git 忽略的 `runs/full400_rgb/`、`runs/full400_semantic/` 与 `runs/full400_semantic_moments_cross/`。

## 结果

| 评估集合 | RGB PSNR (dB) | SSIM | LPIPS | 语义前景 mIoU | 语义全类 mIoU | 视角数 | 标注视角数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全部 400 张训练视角 | 31.9617 | 0.914645 | 0.212959 | 0.987054 | 0.989404 | 400 | 300 |
| 原 50/41 相机诊断 | 32.0418 | 0.913451 | 0.214693 | 0.987677 | 0.989909 | 50 | 41 |

原 50/41 相机在全量版本中已经参与训练，所以第二行是训练重叠诊断。它不能与 350/50 结果做无偏优劣结论；论文主表仍应使用原 350/50 检查点和原始评估脚本。

评估产物由 [evaluate_full400_diagnostic.py](../scripts/evaluate_full400_diagnostic.py) 生成，包含逐视角 RGB、语义 mask、混淆矩阵、边界 F1 和运行时间：`runs/full400_evaluation/all400_fit/metrics.json` 与 `runs/full400_evaluation/original50_diagnostic/metrics.json`。

## 解释与限制

全量训练适合作为提交或部署模型：它利用了全部 300 张语义标注和 100 张无标注 RGB。但由于没有新的独立相机，当前实验不能回答“增加 50 张是否提升未见视角泛化”。要回答该问题，需要固定一个不参与训练的新采集集，或在论文中继续使用 350/50 协议。

该实验验证的是 RGB—语义单场及其交叉矩精修路径。最终多场部署还包含独立的来源增强 RGB 场和 MCMC RGB 场；本记录不把这两个外部场的性能冒充为全量训练结果。
