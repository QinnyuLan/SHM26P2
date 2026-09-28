# 固定类别边缘的两列耦合：独立 helper 契约

[semantic_mass_constraint.py](/home/sky/workspace/SHM2026/src/bridge_rgs/semantic_mass_constraint.py) 是供候选审查的数学组件，**未接入模型或训练，没有采用或性能结论**。它实现标准 constrained categorical coupling，不是新求根算法。固定 qbar 是此约束的实质；若把 qbar 一起学习，不能将等价重参数化称为新的质量约束方法。候选的研究边界仍见 [原候选文档](/home/sky/workspace/SHM2026/docs/semantic_partition_mass_constraint_candidate.md)。

接口为 `mass_constrained_endpoints(qbar, h, a)`，返回 `q_in, q_out, allocation, complement, root, diagnostics`。qbar 与 h 是末维 K≥2 的 FP64 张量，a 为 FP64 张量或实数标量，前导 batch 维广播、device 相同。qbar 必须固定（`requires_grad=False`）、有限非负、已归一化；h/a 可微。该函数不隐式归一化、夹概率、增加地板，也不更改输入。

令 `r=sigmoid(h+lambda)`，求唯一根 `sum(qbar*r)=a`，输出

`q_in=qbar*r/a`，`q_out=qbar*sigmoid(-h-lambda)/(1-a)`。

两端 simplex 且 `a*q_in+(1-a)*q_out=qbar`；零 qbar 类别始终为零。负 sigmoid 单独求值，避免 `1-sigmoid(z)` 在正尾饱和后丢失小概率。实现用 `exp(-logaddexp(0, ±z))`，没有 softplus 人为截断。h 先减第一类坐标以减少公共偏置消减，返回的 root 仍是原 h 对应的 lambda。

求根使用预定二分法：以 `log(a)-log(sum(qbar)-a)-max/min(centered_h)` 为包根区间，再各扩 2 个 logit 单位；最多 192 次。大于半质量时使用负 sigmoid 的较小列计算等价根残差。custom backward 使用

`D=sum(qbar*r*sigmoid(-h-lambda))`，
`d lambda/d h_k=-qbar_k*r_k*sigmoid(-h_k-lambda)/D`，`d lambda/d a=1/D`。

端点保留 h/a 的显式计算路径，再叠加根的隐式导数。不是 detach root 后忽略约束；二分分支本身不反传。**只支持一阶梯度：全部五个可微输出都经一阶专用 identity 包装，在 backward 检测到 `create_graph=True` 时立即报错，禁止建立高阶梯度图。** 不能仅依赖根内部的 `once_differentiable`：那样端点除法和 root 的显式中心化路径仍可能返回不完整二阶值；此漏洞已修复，没有新增二阶支持。

数值边界在测试前固定，ε 表示 FP64 machine epsilon：

| 项目 | 条件及处理 |
|---|---|
| 输入归一化 | `abs(sum(qbar)-1) ≤ 64ε`，且 `0<a<1`、`a<sum(qbar)`。不修正实际 qbar 总和。 |
| 根精度 | 残差除以较小列质量 `min(a,sum(qbar)-a)` 不超过 `64ε`。 |
| 导数条件 | D 必须有限且严格大于 `1e-14`。不以 clamp/damping 伪造梯度。 |
| 输出闭合 | 两端 sum 与 1 的绝对误差≤`256ε`；逐类质量相对 qbar 误差≤`128ε`；零质量严格为零。 |
| 范围与故障 | 非有限、负端点、超出上述容差、包根/求根失败均整次显式失败；不剔除行或修补。`a` 的 bool 类型被拒绝。 |

`0<a<1` 与有限 h 并不足以保证有限精度可计算。近端点会放大输入 qbar 的归一化舍入；例如输入 sum 偏 `2ε` 虽在输入容差内，也可能在 a=`1−1e−10` 的输出闭合检查失败。极小类别在极端 h 下仍可能下溢；diagnostics 明列正 qbar 对应的零端点/分配数，不能声称任意微小质量均可辨识。没有悄悄限制或修正这些数据以使测试通过。

**近端点的一阶导数也有精度限制。** forward 闭合与 D 门不保证端点对 a 的导数有统一绝对/相对精度：显式除法项与隐式根项可能相消。独立 70 位参考检查中，qbar=`[.125,.25,.375,.25,0]`、h=0 、线性系数 cin=`[.3,−.2,.1,1,.7]`、cout=`[−.4,.5,.9,−.2,1.1]` 的端点目标理论 a 导数为 0，而 a=`1e−10`/`1−1e−10` 时 FP64 实际约为 `−8.58e−6`/`−1.91e−6`；`sum(q_in)` 的导数也可出现约 `3e−5` 偏差。同期两列重构质量的梯度残差仍≤`3.30e−15`，forward 质量残差≤`5.55e−17`，故不能将单端点导数消减误写成同量级质量泄漏。近端点测试只证明该固定例前向闭合、导数有限；独立有限差分精度合同是在正常 a 范围测试。这里如实报告限制，不改求根、门或概率。

32 项 CPU 测试覆盖 h=0 恒等、公共平移及 lambda 梯度、类别置换/广播、固定前向安全例 a=`1e−10`/`1−1e−10` 与 q=`1e−200`、正尾小概率及梯度、SciPy 独立求根与 h/a 有限差分、Torch gradcheck、a=.5 非均匀 h 必须真正求根、饱和 D 的拒绝、无效输入；高阶入口测试覆盖 q_in/q_out 非线性目标、allocation/complement、root.square() 对 h/a 的所有路径。使用项目 uv 环境、禁 CUDA；没有真实数据、模型或 runtime 性能测量。diagnostics 会同步读取标量，因此也不作吞吐量承诺。
