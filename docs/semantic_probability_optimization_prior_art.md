# 概率合成、责任因子与优化：有限先例核查

2026-09-27。仅阅读已完成审计、历史冻结源码及下列三篇一手文献；未读取正在运行的 eps-only 产物，未读新像素、运行 GPU 或修改实验。**已有明确的“先渲染 feature、后分类”方案；未核到能把本项目 raw 低分归因于责任饥饿、或证明 categorical 自然梯度可解决它的同对象定量对照。** 这是有限检索结论，不是原创性证明。

| 一手来源及确切位置 | 已有明确方案 | 与本项目的差别及证据界限 |
|---|---|---|
| [Gaussian Grouping，§3.2.2、Eq.1/3](https://arxiv.org/html/2312.00732v1#S3.SS2.SSS2)；[作者 train.py](https://github.com/lkeab/gaussian-grouping/blob/main/train.py#L76-L96)，`render_object → classifier → CrossEntropyLoss` | 先 alpha 合成 16 维 identity feature，再线性分类与像素 CE；另有点上分类分布的邻域 KL。 | 本项目 raw 是先逐点 softmax 再合成概率，顺序不同。其 §4 的 mask association、feature 维数和 3D 正则消融，并非固定同一场的 softmax 前后位置配对；不能把论文整体收益归因于这一顺序。 |
| [FAST-Splat，§4、Eq.3](https://arxiv.org/html/2411.13753v2#S4)；§5.1–5.2、Appendix C | 渲染 semantic feature 后做仿射变换/softmax，以加权 CE 联合训练。 | Tables 1–3 比较完整方法，消融主要比较闭集/开放集检测器；没有核到“概率先合成 vs logits 先合成”的同 W 单因素定量结果。它说明该读出路径已有先例，不说明我们换路径必然改善。 |
| [Variational Bayes Gaussian Splatting，§3.1–3.2、Eq.18–22](https://arxiv.org/html/2410.03592v2#S3.SS2)；[作者代码](https://github.com/VersesTech/vbgs) | 对位置与 RGB 的生成混合模型做 CAVI：估计潜在 Gaussian 归属，再累加责任加权充分统计量，更新共轭后验。 | categorical 变量是“样本属于哪个 Gaussian”，并非“Gaussian 属于哪个语义类”；拟合目标也不是固定透射率的语义渲染 CE。自然参数闭式更新不等于 categorical 自然梯度。其 RGB/持续学习对照不能证明本项目的语义 EM 或 simplex 优化收益。 |

以下为针对当前模型的代数推导，非上述论文的性能结论。固定几何/透射率，含固定背景项，令 `q_i=softmax(z_i)`、`p_r=Σ_i W_ri q_i + W_r,bg q_bg`。概率 floor 不活跃时，单像素加权 CE 满足

\[
\nabla_{z_i}L_r=c_y\underbrace{\frac{W_{ri}q_{i,y}}{p_{r,y}}}_{\text{当前目标类的后验责任}}(q_i-e_y).
\]

因此 `q_i,y` 极小时，正确类责任可能小；但这不等于该 Gaussian 对所有 rays 都无梯度。对直接 simplex 坐标，欧氏导数是 `∂L/∂q_i,y=−c_y W_ri/p_r,y`，仍须满足行和与非负约束，不能把无约束坐标导数直接当可执行更新。逐点 categorical Fisher `diag(q_i)−q_i q_iᵀ` 也不是观测到的合成像素分布的完整 Fisher：后者经同一 ray 耦合多个 Gaussian。局部 Fisher 预条件、EM、直接 simplex 优化及 feature Adam 是不同算法，不能互相视为精确等价。

若改为 `p̃_r=softmax(Σ_i W_ri z_i+固定背景logit项)`，相同假设下导数变成 `c_y W_ri(p̃_r−e_y)`，没有逐点 `q_i,y` 责任因子；同时前向概率族也变了，不是只修 optimizer。线性 decoder 与合成在正确处理 bias/背景时可以交换，**softmax 与合成一般不能交换**；未归一前景权重下，`AΣWf+b` 与 `ΣW(Af+b)` 的 bias 差也必须区分。

Adam 的 `−η m̂/(√v̂+ε)` 说明，小梯度的影响取决于同一参数的历史二阶矩与 eps；仅比较梯度绝对值不能证明停滞。保持整段梯度仅作常数缩放时，eps 可破坏近似尺度抵消，但实际责任因子随点、类、ray、训练步变化，不能由此推出 Adam 会完全消除它。本次正在运行的 eps-only 配对只检验 partition 新 field groups 的优化尺度，不能追溯解释旧 feature 训练——[旧 raw 审查](/home/sky/workspace/SHM2026/docs/raw_semantic_optimization_review.md)已核实原 feature Adam 是 eps=1e−15，decoder/head 才是 1e−8。

已有反证必须保留：[259 TRAIN 单轮 assignment](/mnt/data/SHM2026/runs/raw_semantic_assignment_v1/execution_receipt.json)中，近似 EM 的真实 CE 下降，但 raw 五类仅 +0.063857pp、索 +0.040347pp，冻结 final 五类 −0.565557pp；259 步 Adam 的 CE 下降更大而索 −0.364870pp。两臂约束与更新成本不同、均未证收敛，因此既不能宣布 EM 已解决瓶颈，也不能宣布所有概率空间求解均无效。[结果与限制](/home/sky/workspace/SHM2026/docs/raw_semantic_assignment_results.md)

目前能确定的是读出函数、训练目标和 optimizer 尺度要分开解释。raw 约79%/索约31%与 refined 94%+同时改变了 RGB/空间上下文和函数容量；该差距本身不能识别责任因子、几何不足或 softmax 顺序的因果贡献。本 note 不追加训练、阈值或方法采用建议。
