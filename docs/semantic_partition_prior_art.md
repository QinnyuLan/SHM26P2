# 共享可见性的 Gaussian 内部语义带：先例与数值合同

状态：独立 CPU 验证器已实现；完整固定合成数值实验须由根任务冻结源码和计划后执行。本轮不读取真实场景、相机、RGB、标签，不训练，不调用 GPU。它检验公式与实现，不报告语义质量或新颖性已成立。

## 直接重合与剩余问题

| 一手来源 | 已覆盖的技术 | 本候选可以区分的范围 |
|---|---|---|
| [3D-HGS, CVPR 2025](https://arxiv.org/html/2406.02720v4)，§3.2 Eq. 8–9、§3.3 | 内部平面分割 Gaussian，沿视线解析积分为二维 Gaussian 乘条件 Gaussian 的 erfc；学习方向并组合两半的透明度。 | 此处保持原 RGB 密度、形状、透明度与完整合成权重，只改变每个 primitive 内的类别概率。解析积分、可学习内部平面并非新技术。 |
| [XClipGS](https://arxiv.org/html/2608.07760v1)，§3.2 Eq. 4–5、Proposition 1 | 任意偏移平面的 Gaussian clipping；Schur 条件方差与 CDF 系数，明确仿射 EWA 下的单 primitive 精确性。 | 两个半空间之差就是 slab。双边 CDF 和下面的质量恒等式不能称为新数学。XClipGS 的裁剪会改变可见性，本候选拟固定原光学权重。 |
| [Neural Texture Splatting, SIGGRAPH Asia 2025](https://arxiv.org/html/2511.18873v1)，§4.1 | 在 Gaussian 局部坐标中查询空间变化的 RGBA 纹理；用局部纹理增加 primitive 内部表达。 | primitive 内非恒定属性本身已有。必须与相同参数、只在条件均值查询的 point control 区分，不能仅胜过恒定类别就声称积分贡献。 |

当前可检验的问题仅为：**共享光学可见性时，对局部类别带进行条件协方差积分，能否比相同局部语义参数的点查询提供可复现增量？** 尚未进入真实训练，也未证明任务需要这类表达。未查得完全同型的语义应用不等于全领域首次；目前没有已验证 novelty。

## 被验证的数学对象

令无量纲局部坐标 \(\xi\sim N(0,I)\)，仿射投影 \(\delta=P\xi+\epsilon\)，\(\epsilon\sim N(0,E)\)。\(P\) 的单位为图像长度，\(E\) 为图像长度平方；\(n\) 是单位向量，\(b,w,\tau\) 用局部标准差单位，\(w,\tau>0\)。

\[
C=PP^T+E,\quad m=P^TC^{-1}\delta,\quad V=I-P^TC^{-1}P,
\]
\[
g(\xi)=\Phi((n^T\xi-b+w)/\tau)-\Phi((n^T\xi-b-w)/\tau),\quad
q(\xi)=q_{out}+g(\xi)(q_{in}-q_{out}).
\]

三个模式始终共享 \(P,E,n,b,w,\tau,q_{in},q_{out}\)：integrated 使用条件均值和 \(\tau^2+n^TVn\)；point 使用相同均值、只保留 \(\tau^2\)；marginal 使用无条件均值 0、方差 1，因此是常数类别混合。marginal 不是额外拟合的最佳 constant classifier。

连续完整平面上，投影 Gaussian **归一化后的**类质量满足
\(\mathbb{E}_{\delta\sim N(0,C)}[\bar g(\delta)]=\mathbb E_{\xi}[g(\xi)]\)。这是条件期望的全期望恒等式。它不保证绝对像素总质量（含 \(\sqrt{\det C}\)）、有限画布或遮挡后类质量不变；也不消除透视线性化、截断、alpha 限幅、排序和 early termination 的误差。\(V\) 是单 Gaussian 内部的条件方差，不是已校准的相机位姿不确定性。

## 冻结数值范围

入口为 `scripts/audit_semantic_partition.py`；根任务计划须绑定该入口、实际导入 `semantic_partition.py`、`SPEC` 和输出目录。输出非空时拒绝执行，失败保留，不更改阈值重试。正式矩阵为固定 seed 20260927 的 48 个组合：8 种投影/方向/AA 情况 × 6 种带位置与宽度，包含斜视、近零/近先验条件方差、旋转、强 AA、相关噪声、正负尾部和薄带。

- 独立 NumPy 条件标量分布，加 SciPy `quad` 积分未积分软带；`epsabs=epsrel=1e-11`，Torch CPU FP64 闭式值的绝对误差门 `1e-8`。
- 二维屏幕 Gaussian 的 512×512 Gauss–Hermite 点积分。integrated/marginal 检查归一质量不变；point 对其自身分布的解析期望核验，偏离 prior 只描述，不能预设 point 失败。
- 每个模式检查概率范围、simplex、`q_in=q_out` 恒等、固定完整 W 和残余 class-0 背景，以及图像单位 0.5/2 与世界长度单位 0.01/100 的一致性。
- 10 个预定 case（索引 0/1/7/13/19/25/31/37/43/44，覆盖全部 8 种投影的非零偏移）× 3 mode × 16 个局部参数坐标的梯度：单位法向用切空间图，AA 协方差用 Cholesky 图。固定五点中心差分 `h=1e-4`；差分来自独立 NumPy 闭式函数（其值另以 quad 核验），不是把 quad 误差放大后当导数。`|analytic|>1e-7` 用相对误差 `1e-4`，其余只用绝对误差 `1e-8`。
- 总内部限时 180 秒，4 CPU 线程，无 CUDA 初始化。小型 pytest 只核验证器接线/计算合同，不代替根任务冻结后的完整结果。

数值通过只允许说上述固定合成条件下实现吻合。后续若进行真实语义训练，应固定同一 RGB 几何、W、参数预算和监督比较 integrated / point / constant；这些实验尚未在此获结果，不能用人为构造的数据训练收益代替泛化证据。
