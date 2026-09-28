# 固定可见性下 EM 与 FW 的数学边界

2026-09-27。仅数学说明，不是新试验计划。固定 `W≥0`、背景和正噪声，令 `p_rc=b_rc+Σ_i W_ri q_ic>0`、每行 `q_i` 在类别 simplex，目标 `F(q)=−Σ_rc a_rc log p_rc`。逐view归一化及类别权重并入固定非负 `a`；`δ=5e−7` 可并入 W 的比例和 b，不改变以下推导。

旧点的责任为 `z_ric=W_ri q_ic/p_rc`，背景责任为 `b_rc/p_rc`。Jensen上界中与新q有关的部分为 `−Σ_ic u_ic log q'_ic`，其中

`u_ic=Σ_r a_rc z_ric=q_ic(−∂F/∂q_ic)`，`U_i=Σ_c u_ic`。

当 `U_i>0`，标准归一M步是 `q_EM,ic=u_ic/U_i`。精确算术满足

`F(q)−F(q_EM) ≥ Σ_i U_i KL(q_EM,i || q_i) ≥ 0`。

固定背景的责任不更新；它不妨碍单调性。要求完整目标与W固定、a非负且固定，不能额外混入q相关clamp、归一化或自适应权重。这里是普通Jensen/EM，不是新优化公式。

- **零锁：** q的零分量在EM中保持0；正像素噪声不能解除它。`U_i=0`时保留该行安全，但不能直接称为“未观察”：也可能有用类别都被当前零概率锁住。只有所有相关W为0才是结构性无观察。EM固定点或一次M步都不自动证明全simplex最优。
- **FP32：** 微小责任、下溢及归一误差可能破坏精确保证。跨view可以FP64累计，但需以实际q32端点的完整F验证不升；静默加地板/裁剪后不能仍称精确M步。数值近似梯度也不提供严格gap证书。
- **FW补充：** 用全梯度g取每行最小梯度类别的顶点s，`d=s−q`，初始斜率为 `g·d=−G(q)`。沿线目标凸，包含零步长的有界线搜索可接受下降步，并向原先为0的类别注入质量。EM与FW各步均实际不升时，组合也不升；它减少对任意PG步长的依赖，但不能消除稀少概率曲率或保证有限预算内收敛。

[旧一次EM](raw_semantic_assignment_results.md)及[本次未收敛PG诊断](raw_simplex_fullbatch_results.md)均保留各自结论；本说明不追认收敛、不放行训练，也不构成学术新颖性依据。固定W消除assignment欠优化本身已有[FlashSplat等直接先例](simplex_conditional_capacity_prior_art.md)。
