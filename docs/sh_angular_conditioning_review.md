# TRAIN 角度覆盖约束 SH：候选优先级审查

2026-09-27。仅文献与现有代码审查；未实现、未设实验、未运行 GPU。**建议保留为次级工程候选，优先完成已锁定的完整外观目标诊断；目前没有它能解决浮层或提高精度的证据。**

现有证据不支持直接归因于角度外推：两次固定几何颜色精修的组合目标均在 **350/350 TRAIN** 视角恶化，native 平均 .02906→.03983、original .03671→.04666，不能只解释为未观测方向过拟合。[审计](appearance_train_objective_audit.md) 尚未区分完整 L1/SSIM、有限步长与优化路径；[40-render 诊断](appearance_actual_objective_diagnostic.md) 仍待执行。另一方面，[浮层审计](floaters_and_split_audit.md) 已指出前层真实 opacity 遮挡；SH 收缩不改变 transmittance，最多改变遮挡层颜色，不能恢复被挡住的后层信息。

只采用两篇已读方法正文的一手文献：

| 最接近工作 | 已覆盖内容与本候选的剩余区别 |
|---|---|
| Papantonakis 等，*Reducing the Memory Footprint of 3D Gaussian Splatting*，2024，§4.2（[正文](https://arxiv.org/html/2406.17074v1#S4.SS2)） | 已按 TRAIN 视角、平均 transmittance 加权的颜色方差及删高阶后的颜色差异，为每个 Gaussian 选择 SH 阶数。其目标是压缩并尽量保真，不能据此声称精度提升。本候选改为基底 Gram 的弱特征方向，而非整阶截断；这是不同选择准则，不足以构成创新证据。 |
| Galappaththige 等，*Predictive Photometric Uncertainty in Gaussian Splatting for Novel View Synthesis*，2026 预印本，§3.2–3.4（[正文](https://arxiv.org/html/2603.22786v1#S3)） | 已将完整合成权重与 SH 写成线性系统，并用球面 L2 先验约束未观测方向。它拟合的是额外不确定性通道，冻结原 RGB；因此不是颜色收缩的直接效果验证，但“低观测方向＋SH 先验＋线性可识别性”不能作为新概念。 |

**矩阵必须说明是什么。** 固定几何/opacity/camera，令 \(B_{iv}\in\mathbb R^{16}\) 为 SH3 基底，\(w_{vpi}=T_{vpi}\alpha_{vpi}\) 为实际完整合成贡献。未触发颜色 clamp 的局部加权 L2 模型中，单色通道正规矩阵的对角块是

\[
H_{ii}=\sum_v\left(\sum_p m_{vp}w_{vpi}^{2}\right)B_{iv}B_{iv}^{\mathsf T}.
\]

用视锥计数、Gaussian opacity 或 \(\sum_p w\) 代替平方贡献，只得到角度覆盖代理。跨 Gaussian 的 \(H_{ij}\) 仍含重叠前后层的颜色补偿退化，块对角谱无法识别这种联合不可辨识性。这里也不是生产 L1/SSIM 的精确 Hessian。当前 gsplat 在 SH 求值后执行 `clamp_min(color+.5,0)`，局部信息还需按通道计入其活跃导数；颜色跨过 clamp 后固定 Gram 不再等于局部曲率（本地 `gsplat/rendering.py:495–525`）。

**低成本边界。** TRAIN 相机与冻结点的位置足以在 CPU 分块计算视锥角度 Gram，不读取 RGB/VAL，也不改变场；这只能判断必要的角度退化。现有渲染返回的 radii/alpha 并非逐点 \(w^2\)。gsplat 已有 `rasterize_to_indices_in_range` 交点接口，可支持后续完整遮挡统计，但仍需还原与核对合成、抗锯齿补偿和截断，不能声称现成免费可得。498,136 点的一个完整 FP32 16×16 矩阵约 486 MiB，分块可减峰值，不能消除遍历有效贡献的成本。没有测得运行时间或实际角度谱。

**最大风险与优先级。** 弱方向收缩是标准各向异性正则：收缩现有系数可能抹掉稀有但真实的视角效果；只限制新更新则不能修复已有错误颜色。全 16 维特征向量可能混入 DC，盲目向零收缩会改变已观测均色；完全不可见点更没有数据可确定“正确颜色”。固定几何下还可能只是给浮层涂上更平滑的颜色。只有后续证据表明颜色漂移确实集中在弱观测模式，而可识别模式与完整 TRAIN 目标优化正常，才值得提高其优先级；条件数大本身不是误差成因或收益证据。本轮不追加实现或试验，也不据有限两篇检索声称不存在同型先例。
