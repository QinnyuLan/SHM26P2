# 固定共享可见性的语义可行性：有限先例核查

2026-09-27。使用 academic-researcher 工作流程；只读论文、作者代码与本项目已有诊断，不运行 GPU、不改训练。问题限定为：在已训练场的完整合成权重固定时，当前错误是否仍可由更好的、跨视图共享的 Gaussian 类别分配消除？它服务于可靠性和结构分配归因，不把标注与物理表面的差异作为论文主线，不增加语义支撑面。

**最重要的结论：FlashSplat 已直接覆盖“固定 alpha/transmittance 后，通过线性规划消除语义 assignment 的迭代欠优化”这一大部分动机。** 当前候选最多留下逐射线共同 margin 的条件诊断，以及将条件最优解的敏感度用于结构决策的待验证差异。标准 LP、概率/特征渲染、类别偏置和语义冲突增密都不能单独作为创新。

## 六项最相关的一手对照

| 工作及已核实位置 | 直接重合 | 与本次有限诊断的区别 |
|---|---|---|
| **FlashSplat，ECCV 2024**：§3.2，Eq.2–7；§3.3、附录8.3 | 冻结所有合成权重，把二元 mask 的总 L1 误差写成 ILP，并化成每 Gaussian 的贡献加权投票；多实例扩展也处理一个 Gaussian 对多个对象的贡献。 | 其目标是可分的总误差；本候选是所有指定 ray 同时满足类别间 margin 的耦合约束。不能把“先求最优 assignment”重新命名为贡献。[论文](https://arxiv.org/html/2409.08270v1) |
| **ObjectGS，ICCV 2025**：§3.3，Eq.4–7 | 讨论连续特征混合的歧义，使用 one-hot ID 在完整场中 alpha 合成，保留遮挡，再以 CE 约束几何。 | 先分配 anchor ID，子 Gaussian 继承；未在所核查的方法中先求所有独立 simplex assignment 的共同 margin 上界。[论文](https://arxiv.org/html/2507.15454v1) |
| **GradiSeg，2024预印本**：§3.3–3.4，Eq.2、Algorithm 1 | 明确指出单 Gaussian identity 被边界两侧推向不同方向；累计 identity 梯度后分裂，随后让子 identity 分别适应。 | 当前梯度/阈值是结构触发信号；未消去当前 assignment 的优化状态，也不提供选定 rays 的不可分证书。[论文](https://arxiv.org/html/2412.00392v1) |
| **SAGD，v4，2025**：§III-C–E，Eq.5–12 | 检查 Gaussian 投影长轴端点是否跨 mask 边界，分解并移动/收缩 Gaussian；跨视图中心投票决定类别。 | 是几何边界构造和投票，不是完整 compositing 矩阵上的联合最优分类。语义边界驱动分裂本身已被覆盖。[论文](https://arxiv.org/html/2401.17857v4)、[作者代码](https://github.com/XuHu0529/SAGS) |
| **VALA，2025预印本**：§3、§4.1–4.2，Eq.5、9–13 | 显式使用 marginal contribution αT，按贡献选择可见 Gaussian，再稳健聚合多视图语言特征，处理遮挡引起的错误分配。 | 筛选与聚合不是联合 margin feasibility；提取 W、贡献门控、去除低贡献监督都不是本候选独有。[论文](https://arxiv.org/html/2509.05515v1) |
| **BEA-GS，2026预印本**：§3.3，Eq.2 | 先按实际贡献给 Gaussian 固定类别，再惩罚向错误类别像素贡献的质量，由此更新位置、形状和 opacity；另有不可见区域约束。 | 类别固定，不是每次先对 assignment 求条件最优。本候选也不采用其 occupancy proxy 或未观测区域补约束。[论文](https://arxiv.org/html/2605.09662v1) |

FlashSplat 的作者实现也与论文主链一致：`objremoval.py::render_set` 累计 rasterizer 的 `used_count`，`multi_instance_opt` 比较每对象贡献与其余对象贡献；这不是仅依据摘要判断。它的独立 binary 多对象集合允许重叠，不等同互斥五类 simplex；其 novel-view 对象子集渲染也不能替换本项目的完整场遮挡算子。[作者代码](https://github.com/florinshen/FlashSplat/blob/master/objremoval.py)、[渲染入口](https://github.com/florinshen/FlashSplat/blob/master/gaussian_renderer/__init__.py)

以上是方法级定向核查，不是全领域优先权证明。未在这六项方法中发现下面的完整组合，不等于不存在其他等价工作；也未执行它们的代码或复现其分数。没有扩大到通用凸优化或大规模文献综述。

## 条件 oracle 的准确边界

以下是本项目候选的数学定义与分析，不是上述论文的结论。令固定渲染器对 ray r、Gaussian i 的实际贡献为 W_ri≥0，残余背景 t_r=1−Σ_i W_ri。对于五类独立变量 q_i∈Δ⁴，定义

\[
p_r(q)=\sum_i W_{ri}q_i+t_r e_{\rm bg},\qquad
\gamma^*(W)=\max_{q,\gamma}\gamma\quad\text{s.t.}\quad
p_{r,y_r}(q)-p_{r,c}(q)\ge\gamma\quad(\forall r,c\ne y_r).
\]

这是标准 LP。与总 L1 的可分投票不同，其约束要求选定 rays 同时可分。**低共同 margin 不是“多少像素必然错误”的界**：一个冲突或误标 ray 就能决定最坏值；小正 margin 也不等于不可行。可靠的 γ*≤0 只排除该固定概率合成模型在这些 rays 上同时取得严格正 margin，不能推出全场几何错误、VAL 上限或改变结构必有益。正 margin 则提供这些 rays 的可行 assignment，不保证现训练会得到它、不保证泛化。

当前 `model.py` 先对每 Gaussian 的 decoder 输出 softmax，再合成；与 `softmax(ΣW·logit)` 或“先合成 feature 再分类”不是同一个模型。后两者不受该 simplex LP 上界约束；改变算子本身也不作为新贡献。当前16维 feature 经5类线性 decoder，若四个类差方向满秩，任意内部 simplex 概率本已能由独立 feature 表示；LP主要消除当前优化/正则路径，而非必然提高 feature 维度的表达能力。2D refiner 的图像相关输出同样不在此 oracle 的可行域内。

W 必须来自所用渲染器完整的排序、AA、alpha 限幅/阈值和提前终止语义；先验证 `W·q_current+t·e_bg` 复现 raw 输出。当前实现还做1e−7下限和行归一化，需单列其误差；argmax 次序与可靠正 margin 不能依赖数值边界。截掉 top-K 外贡献并固定其类别会缩小可行域，不能据此证明完整场不可行；应保留全部实际贡献，或给遗漏质量最乐观的自由分配上界，并报告 primal/dual 残差与容差。这里的“精确”指锁定渲染模型，不指物理体积真值。

## 若考虑 dual 压力，尚缺什么

设约束的最优对偶权重 λ_rc≥0，Σ_rcλ_rc=1。适当正则性与稳定活跃结构下，最优值对 W 的候选局部敏感度包含

\[
\frac{\partial\gamma^*}{\partial W_{ri}}
=\sum_{c\ne y_r}\lambda_{rc}
\left[q^*_{i,y_r}-q^*_{i,c}
-\bigl((e_{\rm bg})_{y_r}-(e_{\rm bg})_c\bigr)\right].
\]

最后一项来自残余背景，不能漏掉。LP多解、活跃集变化和 rasterizer 离散可见集合变化时，不应把任意返回的 q*/λ 当作唯一、处处存在的几何梯度。dual 只定位哪些 margin 约束限制了当前条件最优值，不能直接定位“错误 Gaussian”，也不证明一次 split/删除的有限反事实收益。把这种标准最优值敏感度与几何导数组合，仍是已有优化工具的应用候选，不是新 LP 或新 envelope theorem。

可能值得保留的研究问题只有：**在 assignment 已充分放宽后，仍受限的共享可见性是否能比当前语义梯度更可靠地指出需要结构分配的位置？** 若以后有低 margin 证据，还需验证它能区分相机扰动与场表达不足，并使有限结构动作在未参与局部拟合的 TRAIN 视图上受益、保留 RGB 和共同预算。当前没有这样的结果；不在本轮新增训练或结构规则。

现有8 TRAIN诊断只表明：固定4-bias CE校准未提高 raw cable IoU（28.4269%→24.4844%），总alpha≤0.5仅覆盖0.5817%的GT拉索像素。因此总低alpha和这一个校准目标不能解释/修复多数错误；它们**不证明** W 冲突、分类容量上限或某种结构动作有效。[本地实测记录](raw_semantic_mass_diagnostic_proposal.md)

建议等待已限定的少量 TRAIN rays 条件诊断。若大多容易取得高共同 margin，应优先承认这些样本未支持“固定 visibility 难分”，不能再以普通语义梯度增密包装相同动机；若出现可靠低 margin，也只把它作为局部共享可见性的归因证据，暂不作新颖性或性能主张。
