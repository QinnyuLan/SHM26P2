# 相机归因的受控配对实验（H1）

## 当前结论

四组受控实验均已完成：干净/轻度扰动相机 × 原始/相机补偿残差。所有实验训练 6000 步，最终均为 177,378 个高斯点，源代码哈希和 50 个原生分辨率验证视角指纹完全一致。

**当前 H1 是空结果：相机归因在干净与轻度扰动条件下都没有表现出可确认的 RGB 优势，噪声条件相对干净条件的收益差异也没有成立。** 三类比较的 PSNR、SSIM、LPIPS 配对 95% 区间全部跨过零。这不是等价性证明，也不能写成“验证了相机鲁棒性提升”。诊断对已注入扰动有响应，但这种中间信号尚未转化为确认的最终性能收益。

## 固定协议

- 训练/验证划分：350 个训练视角、50 个验证视角，训练种子 42。只比较 RGB 重建。
- 对 349 个训练相机施加左乘 SE(3) 高斯扰动；002.png 保持锚定。平移标准差为场景半径的 0.0005（世界坐标 0.0027397683），旋转标准差为 0.001 rad，扰动种子 20260926。
- 所有验证相机与图像、训练图像及标签、干净初始化保持原样；没有复制或修改原数据文件。
- 两组均 6000 步，hybrid 增密、PCA 结构分裂、AbsGrad 阈值/排名、容量日程、前景配额等配置相同。相机参数不做优化，教师与语义损失关闭。训练标签仍用于固定的 RGB 前景加权和初始化的前景容量分配。
- 归因组启用相机残差解释及默认 camera_quality_weighting；raw 组 camera_attribution=false，使残差未经相机补偿且质量权重为 1。本实验比较完整归因策略，不能分离残差补偿与质量加权各自的效果。
- 每个相机条件内的两组唯一显式配置差异为 camera_attribution 与输出目录；clean 与 mild 对应组只改变 manifest 和输出目录。四组都复用第一组 source_snapshot，已核验全部源文件哈希相同、条件内输入文件哈希相同，原始 manifest 与干净初始化的哈希在运行前后未改变。
- 全部 50 个原生分辨率验证视角均纳入；LPIPS 对无效像素使用目标 RGB 填充预测。评价相机保持官方未优化外参。

## 轻度扰动结果

| 方法 | 步数 | 实际高斯数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| 原始残差 | 6000 | 177,378 | 28.322233 | 0.864876 | 0.308863 |
| 相机归因 | 6000 | 177,378 | 28.246317 | 0.864754 | 0.308695 |

差值方向为“相机归因 − 原始残差”；PSNR/SSIM 正值更优，LPIPS 负值更优。

| 指标 | 均值差 | 配对视角 bootstrap 95% 区间 |
|---|---:|---:|
| PSNR | -0.075916 | [-0.202146, +0.040967] |
| SSIM | -0.000122 | [-0.000324, +0.000082] |
| LPIPS | -0.000168 | [-0.000561, +0.000237] |

## 干净相机结果

| 方法 | 步数 | 实际高斯数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| 原始残差 | 6000 | 177,378 | 29.052841 | 0.885459 | 0.264117 |
| 相机归因 | 6000 | 177,378 | 28.981320 | 0.885553 | 0.264458 |

| 指标 | 归因 − 原始残差 | 配对视角 bootstrap 95% 区间 |
|---|---:|---:|
| PSNR | -0.071521 | [-0.146392, +0.005244] |
| SSIM | +0.000094 | [-0.000055, +0.000260] |
| LPIPS | +0.000341 | [-0.000237, +0.000910] |

干净条件的 PSNR 与 LPIPS 点估计略差，SSIM 略好，所有区间跨零。未发现归因策略带来的通用重建收益。原始残差的 PSNR 从干净条件的 29.052841 降至扰动条件的 28.322233，相机归因则从 28.981320 降至 28.246317；已注入扰动对两种策略均造成了约 0.73 dB 的下降。

## 是否专门减少了噪声影响

在相同视角的四组预测上计算差分中的差分：`(归因_mild − raw_mild) − (归因_clean − raw_clean)`。每次 bootstrap 对四组使用同一套视角索引；PSNR/SSIM 为正或 LPIPS 为负才指向扰动特异收益。

| 指标 | 噪声相互作用差值 | 配对视角 bootstrap 95% 区间 |
|---|---:|---:|
| PSNR | -0.004395 | [-0.138200, +0.126222] |
| SSIM | -0.000216 | [-0.000487, +0.000054] |
| LPIPS | -0.000509 | [-0.001217, +0.000219] |

三个区间均跨零，未验证归因策略专门抵抗本次相机扰动的效果。这一结论限于同场景、单种子、当前轻度扰动和 6000 步预算，未覆盖更长训练或其他噪声强度。

## 相机诊断对扰动的响应

`camera_explained_fraction` 是经实际重新渲染验证、局部相机扰动可解释的加权 RGB 平方残差比例，不是相机误差概率或真值位姿恢复精度。

诊断每 20 步执行一次，仅在 500 ≤ step < 5000 期间生效；日志每 50 步记录最近一次诊断。按实际执行条件推断最近有效诊断步，合并 5000 步之后一直重复的 4980 步统计，得到每条件 **91 个被记录的不同诊断时刻**，约覆盖 225 个实际诊断时刻中的 40%。以下仅是这部分日志的描述统计，不把重复统计或相邻训练时刻视为独立样本，不计算伪显著性。

| 训练阶段 | 每条件记录数 | 干净均值 | 干净中位数 | 扰动均值 | 扰动中位数 |
|---|---:|---:|---:|---:|---:|
| 0.5 倍分辨率，500–899 | 8 | 0.997% | 0.491% | 2.934% | 2.461% |
| 0.75 倍分辨率，900–3599 | 54 | 0.779% | 0.717% | 4.089% | 2.346% |
| 原生分辨率，3600–4999 | 29 | 1.203% | 0.930% | 8.303% | 5.492% |
| 全部有效记录 | 91 | 0.934% | 0.738% | 5.330% | 2.744% |

扰动组在三个阶段都有更大的相机可解释残差，原生分辨率阶段尤为明显。这支持诊断信号对已注入扰动有所响应的有限解释，但不能证明其估计方向与真实注入位姿误差一致；日志没有保存每次估计增量，几何与外观误差也可能被相机方向部分解释。干净与扰动场的训练轨迹已经不同，阶段差异同时受分辨率、SH 阶数和几何成熟度影响。

结合最终性能区间，当前证据更接近“诊断有响应，但其到容量分配与重建收益的转换尚未成立”。不能进一步因果断言问题仅在容量决策：残差补偿与梯度质量加权在此一起变化，尚未分别隔离。逐条有效记录与统计在 `artifacts/pose_stress_mild/camera_diagnostics.json`。

## 统计边界与复现

使用 5000 次视角配对重采样、种子 20260926。区间只刻画本场景这 50 个开发验证视角的重采样变化；相邻视角可能相关，且只有一个训练种子、一个场景和一个扰动强度。这不是独立场景、多训练种子或隐藏测试集的置信结论。并行 GPU 作业影响运行耗时，因此不据此比较训练速度。语义场没有训练，评价文件中的语义输出不用于本报告结论。

- 扰动协议与初始化哈希：`artifacts/pose_stress_mild/protocol_audit.json`。
- 完整配对 RGB 数值：`artifacts/pose_stress_mild/rgb_comparison.json` 与 `artifacts/pose_stress_clean/rgb_comparison.json`。
- 噪声条件交互统计：`artifacts/pose_stress_mild/rgb_noise_interaction.json`。
- 干净相机逐视角评价：`runs/pose_stress_clean_{compensated,raw_residual}/evaluation_native/metrics.json`。
- 原始残差逐视角评价：`runs/pose_stress_mild_raw_residual/evaluation_native/metrics.json`。
- 相机归因逐视角评价：`runs/pose_stress_mild_compensated/evaluation_native/metrics.json`。
- 每组源代码、配置与输入/权重哈希：各 run 下的 `experiment_receipt.json` 和 `source_snapshot/`。

```bash
uv run python scripts/run_experiment.py artifacts/pose_stress_mild/compensated.yaml
uv run python scripts/run_experiment.py artifacts/pose_stress_mild/raw_residual.yaml --source-snapshot runs/pose_stress_mild_compensated/source_snapshot
uv run python scripts/run_experiment.py artifacts/pose_stress_clean/compensated.yaml --source-snapshot runs/pose_stress_mild_compensated/source_snapshot
uv run python scripts/run_experiment.py artifacts/pose_stress_clean/raw_residual.yaml --source-snapshot runs/pose_stress_mild_compensated/source_snapshot
```

以上命令展示本次运行入口；工具会拒绝覆盖已有实验记录，复跑须使用新的输出目录。

当前 30k strong_rgb 实验使用更早的快照，尚未实现 camera_attribution / camera_quality_weighting 开关。不能仅修改 YAML 就把该旧快照当作 raw 对照，本次没有启动这种无效实验，也没有继续进行额外 GPU 训练。
