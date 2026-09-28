# 固定几何的源残差颜色求解：已有方法边界

2026-09-27。仅检索与代数判断，没有实施或训练。当前薄结构固定射线诊断保持优先。

考虑由邻近 TRAIN 源的残差 `e_s = I_s − R_s`，在固定 Gaussian 合成算子 A 下解 `(AᵀA + Γ)δ = Aᵀe`，再输出 `R_target + A_target δ`。这可以处理源像素的多 Gaussian 混合，但不能把 ridge、残差反投影、matrix-free CG 或按目标选少量源视角本身称为新颖点。

最直接的先例是 [Instant Colorization of Gaussian Splats](https://arxiv.org/html/2604.17155v1) §3 和[官方实现](https://github.com/dlieber01/Instant-Colorization-of-Gaussian-Splats/blob/main/colorize_instant.py#L461-L506)。其固定几何、可见性加权小 SH 系统、Tikhonov 与残差迭代已覆盖主要思路。**本项目的代数推断**：在线性颜色与精确伴随假设下，该迭代可写成 `c_next = c + M⁻¹[Aᵀ(y−Ac)−Γc]`，收敛固定点已经满足耦合正则化方程。因此不能用“逐 Gaussian 系统 vs 全部 AᵀA”建立机制创新，CG主要改变求解器。

[G3R](https://arxiv.org/html/2409.19405v1) 已利用渲染梯度将源图信息提升到 Gaussian 并学习更新；[matrix-free second-order GS](https://vcai.mpi-inf.mpg.de/projects/LM-RS/)已有 LM／CG 路径；[IBGS](https://arxiv.org/html/2511.14357v1)则采样靠近中位透射位置的交点，从邻源照片预测图像残差。它们各自的目标、未知量和代价不同，不混报为本桥的新实证。

仍有工程上可检验的差异：源照片像素是混合观测，不能直接视为前景某个 Gaussian 的 radiance。例如 source 像素为 `.2 F + .8 B`，将它直接赋给前景再 alpha 合成，会再次混入背景。完整 A 保留交叉项，但相关列近共线时仍不能唯一辨识前／背景，正则项只选择一个解。

此备选还受剩余噪声／曝光／视变误差、源集合切换和逐新相机求解成本影响。线性声明必须限定固定权重／背景与 clamp 前的颜色，不包括原 SH clamp 或交付量化。排除目标源照片也不撤销底场已训练过该 TRAIN 目标的事实。没有据此新增实验、精度或创新结论。
