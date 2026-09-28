# FP64 simplex 数学工具合同

`src/bridge_rgs/simplex_optimization.py` 仅提供独立 CPU/NumPy 原语；未接生产训练、renderer、数据或完整求解器。这是标准投影梯度与 Frank–Wolfe 条件界，不是新算法或性能结论。

- `project_simplex(values[N,K])`：逐行欧氏投影到 `q>=0, sum(q)=1`；允许精确零，无概率 floor。输入必须是有限 FP64、`N>=1,K>=2`。按最大值平移后，仅将必然不活跃的投影输入限制至 −1 以避免极值溢出，不截断输出概率到正数。
- `validate_simplex(q)`：只检查，绝不自动归一/修补；非负和上界严格，行和 FP64 容差 `1e-12`。直接将 FP32 decoder 输出转 FP64 不保证满足这个合同；如需明确初始化归一化，应由调用者记录这项变更。近可行数值向量不是区间意义的严格可行证明。
- `projected_gradient_step(q,g,step_size)`：返回 `ProjectedStep(proposal,direction,gradient_dot_direction,changed_rows)`。标量步长有限且非负；零步长、全零梯度行、行内完全相同的非零梯度行均逐位保留原 q。其余在乘步长前移除逐行最小梯度这个公共偏置，实数下等价于 `proposal=projection(q-step_size*g)`，避免大共同梯度吞掉 q 的小数；`direction=proposal-q`，点积使用原梯度与实际 FP64 位移。无预条件器、Adam、FISTA、正则或内部线搜索；调用者负责完整目标的 Armijo 检查。中间计算不可表示时明确失败。
- `feasible_direction(q,candidate)` 验证两端后返回差值；实数算术下整条线段均在 simplex。
- `linear_minimization_oracle(g)` 逐行返回最小梯度类别的 one-hot 顶点，平票按类别索引最小者。
- `frank_wolfe_gap(q,g)` 返回 `FrankWolfeGap(vertex,direction,per_row_gap,total_gap,centered_per_row_gap,centered_total_gap,max_simplex_sum_error)`。主值是实际 `sum(g*(q-vertex))`，微负浮点值原样保留、不夹成 0；另报代数等价的 `sum(q*(g-min(g)))` 检查公共偏置消减和近可行误差。两者没有隐含除 N 或 class-weight 总和。

对固定非负贡献 W、固定残余背景，若使用明确的正 affine-noise 概率 `p_tilde=(1-delta)*raw+delta/K`，非负固定样本/类别权重的完整 CE 是 q 的凸可微函数。只有 g 真的是**同一完整目标的梯度**，且 q 可行，实数算术才有 `0 <= F(q)-F* <= gap`。由 CUDA 近似 VJP、随机 minibatch 或不一致归一化产生的量只能称代理 gap；数值前向/反向验证也不自动给出严格区间证书。不得把 `-log(max(p,eps))` 的夹底目标不加条件地当作该凸目标。

合成测试用独立 SciPy SLSQP 约束投影和固定 W 的混合 CE 最优点、解析单行 CE 最优点、中心有限差分、投影下降不等式验证原语；覆盖零/恒定梯度逐位不变、边界零概率、微负 gap 保留及非法输入。它们不证明实际 gsplat 梯度正确，不证明真实优化已充分，也不授权数据实验。
