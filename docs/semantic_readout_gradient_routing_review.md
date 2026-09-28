# 语义读出与梯度分路：实施前审查

2026-09-27。仅 CPU 源码核查及一手文献研究；没有新训练、GPU、参数更新或方法实现。结论是：**保留为待检验的标准梯度路由假设，当前不足以推荐训练或声称创新。** 本次定向检索未定位完全相同的 Gaussian 类差保护与精修残差分路实现，但不是穷尽检索，关键思想已有先例。

## 当前证据不等于历史梯度冲突

已核实 `runs/h3_moments/02_cross/config.json` 与其 experiment receipt：`parameter_scope=refiner_only`、`refiner_field_grad=false`、geometry/RGB 冻结，H3 的 3000 步只训练精修头。因此本次 [4-render 回执](/mnt/data/SHM2026/runs/h3_semantic_readout_diagnostic_v1/execution_receipt.json)不是过去 H3 field 梯度冲突的证据。

固定 TRAIN 002/205 上，移除 Gaussian 特征的类差不可见分量后，RGB/depth/alpha 不变、raw CM 不变，而 final 五类 mIoU 99.016%→84.323%。原索纠正保留 149682/150992，主要变化是新增 148973 个索误报。这只支持固定头对普通 feature 与相关 moment 输入的联合依赖；投影是分布外干预，还改变幅度、相关性、null 分量均值与 GroupNorm 前输入。不能由此证明信息必要性、raw 容量上限、梯度冲突或投影训练有效。

## 假设和三项硬边界

对五类线性 decoder 定义 `D=W[1:]-W[0]`，`P=D†D`、`N=I-P`。固定 decoder 与几何时，真实特征位移满足 `DΔf=0`，便保持每高斯 softmax 概率及 raw 合成概率不变。当前 rank(D)=4 已足以表达五类内部概率；12 维不可见分量不是类别容量缺陷的证据。

| 路径 | final 对 Gaussian 特征的作用 |
|---|---|
| full coupled | 可改变类别概率及精修特征，并经 p3d 更新 decoder |
| Rdetach | final 不更新 field；raw 监督仍可更新 field/decoder |
| 拟议分路 | final 仅更新类差不可见分量；raw 更新类别方向，保留可学习残差特征 |

1. **投影梯度不等于投影实际 Adam 位移。** 坐标式 Adam 的逐维预条件一般与 N 不交换，raw 的行空间梯度也可能被转出行空间。只加 gradient hook 不能声称保持 raw 概率。若使用固定正交基中的两组 Adam，需同基、同优化器的 coupled 控制，排除重参数化混杂；本质相当于 4 维类别与 12 维残差分路，不是新线性代数。
2. **固定初始 D 与持续更新 decoder 存在矛盾。** 当前 decoder 改变后，旧 nullspace 不再保护当前分类；冻结 decoder 则改变原 coupled 的自由度，必须明确且匹配。保护条件还要求对应的 bias、几何和合成权重不被 final 更新。
3. **必须处理全部 final→p3d 旁路。** 现精修头同时读取 p3d 与 entropy，只 detach 最后 `log(p3d)` 基底仍有旁路。普通 feature 和深度—feature moment 的 final 梯度须在 Gaussian 特征源头统一处理，且不能把 raw 与 final 的总梯度一起投 N。raw 应保留独立路径。

## 一手文献与重合边界

- **OGD，AISTATS 2020** 已将新目标梯度投到保持已有输出的子空间，直接覆盖“保护原生读出”的基本动机。[论文](https://proceedings.mlr.press/v108/farajtabar20a.html)
- **Adam-NSCL，CVPR 2021** 将 Adam 产生的候选参数更新投到近似 nullspace；不能把投影原梯度后使用 Adam 当作同一保证。[论文](https://arxiv.org/abs/2103.07113)、[作者代码](https://github.com/ShipengWang/Adam-NSCL)
- **PCGrad，NeurIPS 2020** 已用任务梯度投影处理负迁移。它针对冲突方向，拟议方案则删除 final 的全部类别方向，包括可能有益的方向，约束更强。[论文](https://arxiv.org/abs/2001.06782)、[作者实现](https://github.com/tianheyu927/PCGrad)
- **HarmoGS，2026 预印本** 已在 3DGS 中进行跨视图 Gaussian 属性梯度冲突检测与正交协调；并非固定语义 decoder nullspace，但排除了“首次将正交梯度用于 GS”的宽泛表述。[论文](https://arxiv.org/html/2605.13073v1)
- **Feature 3DGS** 已有 Gaussian feature 与图像 CNN decoder 联合学习；**ViM** 已讨论 logits 看不到的 feature 变化。“概率之外仍有可用特征”本身不是新贡献，也不证明本场不可见分量具有物理语义。[Feature 3DGS 官方代码](https://github.com/ShijieZhou-UCLA/feature-3dgs)、[ViM 论文](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.pdf)

## 继续前的最小必要条件

只考虑前瞻性的 TRAIN 梯度检查，不视为训练授权：固定原头、decoder、geometry，对预先确定的 TRAIN 视图，分别求原实际加权 raw loss、完整 final loss，以及断开全部 p3d 路径后的 final feature/moment 梯度。即使暂时启用 field 求导，也只是研究一个未来更新方向，不能追溯为 H3 原本执行过的更新。

- 剩余 final feature 路径须确实与 raw 梯度存在显著、可测的有害干扰，而非冲突仅来自 p3d 旁路。报告整体方向导数与逐视图值，不能只用负内积高斯占比。
- nullspace 内须有超过数值噪声的可用 final 下降方向；若几乎为零，方案接近 Rdetach。
- 检查移除类别方向是否同时丢失更多有益协同。正交约束不自动保证联合目标更好。
- 核验拟用优化器的实际 `Δf` 和 raw 不变性，而非仅梯度投影残差；同时检验 raw 与真实 final 目标的方向变化。预测 final 端点变化必须使用完整 final 导数，不能用 detach 后的代理导数忽略 raw 更新对 p3d 的影响。

没有稳定、可测的冲突，就缺乏施加该约束的机制依据；即使满足，也只支持局部更新可行性，不能证明 VAL 改善、历史失败原因或学术创新。若未来改正交基/冻结 decoder，必须匹配这些变化的控制，不能把其影响归给投影。
