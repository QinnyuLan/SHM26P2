# Corner v2：固定两阶段训练结果

2026-09-26，修正像素坐标协议后的 RGB 30k 与语义 8k 两阶段均自然完成，终点评测与来源审计通过。**本报告只记录 `colmap_corner_v2` 原生网格结果**，不并入 legacy 成绩表。最终五类 mIoU 为 **94.2093%**，索区 IoU 为 **93.2023%**；未达到五类 95% 目标。坐标修正是工程正确性工作，不作为学术创新或单独性能增益宣称。随后完成的[共同原图网格对照](corner_v2_official_comparison.md)也未支持整体提升，旧模型继续保留。

## 固定实验与评价口径

- 数据为独立的 `artifacts/prepared_corner_v2`，原生 1320×989、seed 42、每 8 个相机留出一个、无 BA。400 个视图中 350 TRAIN / 50 VAL；有标签部分为 259 TRAIN / 41 VAL。旧 prepared 的全部 705 个文件未修改。
- RGB 从新 NPZ 的 60,000 个 TRAIN 三角化点开始，未使用旧 checkpoint；30,000 步与既有 support-constrained split 配置一致。终点为 498,136 个高斯。
- 语义从这一个已完成的新 RGB checkpoint 开始，重置 multiscale64、pyramid-strip、bound6 精修头；259 个带标签 TRAIN 视图、独立 shuffle RNG、seed 42、全原生分辨率 8,000 步。语义场与精修头采用 coupled 梯度，几何、SH、opacity、背景 RGB 与相机冻结。没有 teacher、pseudo、fusion、增密或语义几何损失。
- 两阶段均 `eval_every=0`，save 分别每 5,000 / 2,000 步覆盖滚动 last；没有中间验证选模或额外 step checkpoint。配置在 RGB 开始前锁定，没有根据途中结果改变超参。
- RGB 指标平均于 50 个 held-out 视图；语义 IoU 来自 41 个有标签视图的累计混淆矩阵，包含背景。使用各视图原始、未优化相机与 v2 undistorted pinhole valid 网格。LPIPS 按现有 native 协议将无效像素预测替换为目标值，因此不是全官方原图口径。
- camera-only 预测器只接收相机及渲染证据；评价器可读取 RGB/GT/valid 用于评分，不能声称整个评价进程从未打开 GT。上游官方相机标定使用全体视图，故本实验是给定标定条件下的图像监督留出，不是独立 SfM 泛化测试。

## 原生 v2 终点

| 阶段 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 五类 mIoU ↑ | 前景 mIoU ↑ | 索区 IoU ↑ |
|---|---:|---:|---:|---:|---:|---:|
| RGB 30k | 30.2053319 | 0.903095849 | 0.223966179 | 不适用 | 不适用 | 不适用 |
| 语义 8k，最终精修输出 | 30.2053319 | 0.903095849 | 0.223966179 | 94.209326% | 92.957801% | 93.202324% |
| 语义 8k，原生 3D 场渲染 | 同上 | 同上 | 同上 | 78.680282% | 75.026144% | 30.515391% |

RGB 阶段没有训练语义头，评价文件里的临时语义输出不作为有效语义成绩。原生 3D 场仍是以二维 held-out mask 评分的投影结果，并非三维语义真值评价。最终精修输出使用相机条件下的渲染特征与多尺度上下文，不能将其 94.2093% 称为纯三维点语义准确率。

| 类别 | 最终精修 IoU | 原生 3D 场投影 IoU |
|---|---:|---:|
| 背景 | 99.215426% | 93.296832% |
| 桥面 | 96.316923% | 94.292561% |
| 索区 | 93.202324% | 30.515391% |
| 桥塔 | 93.303790% | 91.694950% |
| 基础 | 89.008169% | 83.601673% |

基础与桥塔是当前最终输出的相对弱项，但这里不据终点选择新的阈值或追加训练。单桥、多轮开发条件下的分数不是跨场景泛化证据。

## 完成与冻结审计

`scripts/audit_corner_v2_stage.py` 对两个阶段均输出 `status=passed`：

1. run receipt 为 completed，实际配置、输入 SHA、全部 source 文件 SHA、30k/8k 终点、checkpoint profile 与 manifest SHA 相互一致；终点张量均有限值。
2. 语义输入 checkpoint SHA 恰为已完成的新 RGB 终点；8 个模型状态项逐位保持相同：means、log_scales、quats、opacity_logits、sh0、sh_rest、background_logits、semantic_prior_counts；training_cameras 也逐位一致，feature_dim、SH degree、scene_scale 相同。
3. 语义终点与 RGB 源的全部 **50 张 RGB PNG SHA 一致**，PSNR、SSIM、LPIPS 逐值相同。
4. 两个 native 评价均为 scale=1、50 RGB / 41 semantic、1320×989，且 v2 fingerprint 与旧 native 不同。

训练耗时截至最后一步：RGB **1,118.76 s**、语义 **737.38 s**；训练峰值 PyTorch allocated GPU 分别 **1.437 GiB / 5.353 GiB**。这两个时间不包含终点评价，也不是独占设备延迟 benchmark。外层 session 17212 / 63153 均在 train 与 native evaluation 完成后自然退出 0；未终止其他进程。完成后 GPU compute 列表为空，磁盘可用 **1,700,515,840 bytes**，已交还给后续固定官方网格评价。没有删除产物。

## 可复核记录

- 训练预先锁定计划：[corner_v2_training_plan.json](../runs/corner_v2_training_plan.json)。数据来源、预算与对应性：[preparation_receipt.json](../runs/corner_v2_preparation/preparation_receipt.json)、[comparison_audit.json](../runs/corner_v2_preparation/comparison_audit.json)、[grid_audit.json](../runs/corner_v2_preparation/grid_audit.json)。
- 两阶段配置：[RGB](../configs/corner_v2_rgb_full.yaml)、[semantic](../configs/corner_v2_semantic_coupled.yaml)。
- 完成与审计：[RGB receipt](../runs/corner_v2_rgb_full/experiment_receipt.json)、[RGB audit](../runs/corner_v2_rgb_full/stage_audit.json)、[semantic receipt](../runs/corner_v2_semantic_coupled/experiment_receipt.json)、[semantic audit](../runs/corner_v2_semantic_coupled/stage_audit.json)。
- 完整逐视图指标和混淆矩阵：[RGB native metrics](../runs/corner_v2_rgb_full/evaluation_native/metrics.json)、[semantic native metrics](../runs/corner_v2_semantic_coupled/evaluation_native/metrics.json)。

| 绑定对象 | SHA-256 |
|---|---|
| 两阶段共同 source tree | `107bfe24f70ba35a4b359c1520e0d97abfd8a59b8209409a720994b880a4f224` |
| v2 manifest | `91b41aeecedc352e4882eb80a1bb635d479ae432251a4c96c85a9dc110cbb327` |
| v2 init NPZ | `544917a1697f9363168ad1632f8f9370f7ff1efa0ef240091501b60e45b158c2` |
| RGB last | `7fac0df69f34cbb8ecefdd32c34d1ffd4ae8b7f41f54293c75056effaa36951f` |
| semantic last | `a77d304f32de4356c1a4608ed5192258b6a9ced4c67a4297809f439481408d89` |
| semantic native metrics | `aac105ebb45fd550f43ec5f7a97d3cf3e9b0afa32f33cbf39d14a8a2f023fc2f` |

v2 native fingerprint 为 `f5c4419d3d81183310ae322b2c6343406a3e00a3e903a5f9a6e3d549bb63fca9`；旧 native 为 `750b9f53046bc104093715c6c26c090837746c467445364584570feb21f10b7b`。两者 target/valid/discrete sampling 不同，且新初始化颜色与语义票也有变化，**不直接跨 fingerprint bootstrap，不把数字之差解释为仅一个坐标修正的因果增益**。另行固定的公共官方原图网格评价应使用各 checkpoint 自身 profile 导出，在同一原始 GT 网格独立报告；它不改变本报告的 native 口径。
