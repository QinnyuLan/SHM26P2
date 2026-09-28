# 固定可见性属性求解与结构容量：有限先例核查

2026-09-27。只查一手论文/作者代码，未读取当前 simplex 运行结果，未使用 GPU。问题严格限定为：固定几何与可见性后直接求每 Gaussian 类别 simplex 上的 CE，并用属性子问题的最优性 gap 或剩余误差决定是否增密、避免将属性欠优化误称为几何欠容量。

**首先必须保留最直接的已有证据：FlashSplat（ECCV 2024）已固定 alpha/transmittance 合成权重，将二元 mask 的总 L1 误差写成 ILP，再化为逐 Gaussian 贡献加权投票。** 它直接覆盖“固定 W 求语义 assignment、消除迭代欠优化”这一核心思路，不能把该动机重新命名为创新。[论文 §3.2 Eq.2–7、§3.3及附录8.3](https://arxiv.org/html/2409.08270v1)；[作者实现 `objremoval.py`](https://github.com/florinshen/FlashSplat/blob/master/objremoval.py) 中 `render_set` 累计 `used_count`，`multi_instance_opt` 比较对象贡献。此前已完成的[本地先例核查](semantic_feasibility_prior_art.md)保留了这一证据，新检索不能将它遗漏。

FlashSplat 的二元 L1 目标可分解成逐 Gaussian 投票；本次互斥五类 simplex 的 `−log(Wq+b)` 在同一像素内耦合多个 Gaussian，通常不能沿用该闭式投票。**这一区别是目标/求解合同的差异，不使“先充分求属性再讨论结构”成为新思想。** 本次有限核查没有找到完整实现“该耦合 CE 的可信全局 gap 再触发增密”的直接证据，但未找到不是首次性证明。下表补充三项近作，不能覆盖或代替 FlashSplat 这一更直接先例。

| 一手来源与准确位置 | 已有内容 | 与当前窄问题的差别 |
|---|---|---|
| [Instant Colorization of Gaussian Splats, v1](https://arxiv.org/html/2604.17155v1)，§3 Eq.4–7、10–15、Algorithm 1；§4.3 | 固定已有场，借渲染颜色导数计算可见性及目标属性均值，逐 Gaussian 解小型加权 LS。语义实验用 L=0 投影 SAM2 二值 mask，再阈值筛 Gaussian。 | 这是明确的固定场属性求解先例。Eq.10 是逐 Gaussian 对目标像素的加权 LS，并非全耦合渲染损失的 simplex CE；正文明确透明混合需另作残差迭代。已核方法没有属性最优性 gap 驱动增密。 |
| [When 3D Gaussian Splatting Recovers Real Surfaces, v1](https://arxiv.org/html/2608.30054v1)，§3.2、§4.2 Proposition 1；Appendix A.2–A.4 Eq.9–15、F.1 | 固定候选 surface 后，最优 appearance 是条件目标在有限 SH 空间的加权正交投影；以最小可达损失分析几何/外观歧义。 | 已有“先消去属性最优值再讨论几何”的明确理论先例。它使用 first-hit 不透明 surface 抽象与 L2/SH，不是多层 alpha 混合的类别 CE；实验证实部分仍采用常规增密日程，没有实现语义内层 gap 控制结构动作。 |
| [SPARE-GS, v1](https://arxiv.org/html/2607.16624v1)，§III-C Eq.12；§IV-A Eq.15–17、IV-C Eq.23 | 将结构预算分配写成边际效用平衡问题；实际用区域位置梯度和可见次数的排名代理调制增密/剪枝。 | 与“最优性指导结构分配”相关，但实际代理不先充分优化固定属性，也不是每点类别凸问题的全局 gap。因此不能把其 KKT 叙述误读为已经剥离属性欠收敛的容量诊断。 |

作者代码进一步核到 [colorize_instant.py 固定提交 1fd119aa8bf682cc4aa5ece5e8bd39055d915025](https://github.com/dlieber01/Instant-Colorization-of-Gaussian-Splats/blob/1fd119aa8bf682cc4aa5ece5e8bd39055d915025/colorize_instant.py)：`collect_solver_statistics`（359–417行）累计可见性/属性，`solve_color_system`（420–433行）解逐点正规方程，499–547行以渲染残差迭代修正。这里没有 simplex 投影、语义 CE 或依据子问题 gap 的增密接线；不能将这个小型 LS 解直接说成 `min_q CE(Wq+t e_bg)` 的全局解。

本项目的解释边界：即使固定 W 的凸目标得到足够小的可信 gap，得到的也是该 W、固定可行域/权重/标签/噪声项下的条件最优性。它并不单独证明真实几何错误、必须增密、mIoU 上限、refiner 最优或新视角收益。若只跑有限步且 gap 未小，只能报告求解进展，不能称剩余误差不可拟合。实际 FP32 算子的有限方向预检也不是全局梯度误差证书；严格 gap 上界仍要求所用梯度符合目标，或另有误差界。当前旧单轮 EM 未解答这一点，但也不构成新方法优越性的证据。

检索边界：本次核查三项近作原文/代码，另补回此前已核实的 FlashSplat，不重新扩大检索。关键词包括 Gaussian splatting + fixed geometry / convex semantic / probability simplex / optimality gap / variable projection / densification。没有扩到教师重训或新优化配方，没有利用当前实验结果调整检索结论。
