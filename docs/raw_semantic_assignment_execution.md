# 单轮原生语义分配诊断执行合同

2026-09-27。已完成 CPU 实现、8 项数学/梯度路由测试和独立审查；GPU 执行须等协调者单独交接。该诊断不修改生产训练或已有模型，也不是新的 EM 方法。数学及现有工作的边界见 [方案](raw_semantic_assignment_proposal.md)。

固定输入是 391a 联合场和 corner-v2 的 259 张有标签 TRAIN；不读取真实 RGB 或任何 VAL 像素。259 名单按名称排序后仅用 NumPy seed42 置换一次，四个阶段复用完全相同顺序。模型与相机来自 391a 的原冻结源码；新增独立数学模块和 runner，不替换任何原 package 文件。

- 共同起点：base 的实际 FP32 类别概率转 FP64 后逐行归一，再以 ε=1e−5 内化；通过固定 decoder 类差矩阵的伪逆实现最小范数 feature 改变量。两臂实际 FP32 features 及实际 decoder qold 必须逐位相同。只记录 base→起点的参数/概率差，不额外测 base 的完整目标，不把初始化算作收益。
- 近似 EM：259 张 E-pass 同时测共同起点目标；颜色 leaf 与场参数隔离，累计实际 qold×负颜色梯度。一轮 water-filling 后仅映回非零责任行；零责任行保留共同起点 features 逐位不变。
- Adam：恢复共同起点，fresh Adam 仅训练 features，259 次更新，lr=.01、eps=1e−15、betas=(.9,.999)。其 raw 捕获不 detach colors，保留 feature 梯度；它不受 EM 的 ε 下界约束，FP32 零概率不作为额外失败门。
- 两个固定终点各另测完整 259 TRAIN。总计 1,036 次 scene render / 2,072 次 rasterizer，259 次颜色 backward、259 次 Adam 更新、一次 M-step。没有 extra sweep、额外样本、VAL、候选搜索或生产 checkpoint。

两臂目标均为固定 δ=5e−7 的 affine-noise raw CE，使用原 TRAIN class weight power .25，每视图有效像素均值再对 259 视图等权。FP64 完成仿射混合、log 和聚合；不对 raw clamp/归一化。残余背景及均匀噪声不分配给 Gaussian。gsplat VJP 存在已观测的尾部透射率重建误差，因此只称“近似 EM 候选”。

唯一数值下降判据预先固定为：共同起点 noise CE − EM 终点 noise CE **大于 max(1e−6, 1e−4×初始 noise CE)**。否则只报告差值并记 unresolved，不追加迭代。通过也仅代表这一固定 TRAIN 目标数值下降，不意味着质量显著提高或采用模型。raw/final TRAIN 混淆矩阵、IoU 和归一概率 CE 均完整报告，哪怕 CE 降而 IoU 降；final 使用冻结 refiner，其变化还混有 feature 表示变化。

完整 JSONL 保存每次实际调用、视图、camera index、raw objective 和 RGB hash。三个评分阶段报告 raw/final pooled CM 和按视图均值 CE，零 alpha 前景数量及噪声责任比例。每阶段核验非 feature 参数/相机逐位不变；每图 RGB 与共同起点一致；finally 恢复所有原始 state tensor 及 flags。仅保存可复核的 NPZ 诊断 features/counts/probabilities，预计约 200 MiB，预算 400 MiB；硬限时 600 秒，失败保留、不重试。

代码：`scripts/audit_raw_semantic_assignment.py`、`src/bridge_rgs/semantic_assignment.py`。CPU 测试覆盖带固定成分的责任公式、零 alpha 有限常数损失与零梯度、water-filling KKT/下界、满秩实现误差、EM/Adam 梯度路径及零责任行逐位保留。协调者随后授权的单次执行已完成，见[独立结果报告](raw_semantic_assignment_results.md)；冻结计划和本合同的阈值均未修改。
