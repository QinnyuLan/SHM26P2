# 固定可见性下的 raw-only assignment：条件性最小方案

2026-09-27。**仅后续proposal，尚未授权执行、未实现GPU训练、未验证收益。** 先完成强参考和固定16-ray v2真实forward诊断；只有v2提供局部可表达性证据后，再由协调者决定是否启动。此方案是标准类别混合优化的诊断工具，不作为新EM、凸优化或创新方法。不新增VAL评价，不改变既有模型、划分或历史结论。

## 数学对象及容易混淆的梯度

固定391a场的geometry、opacity、相机和完整遮挡。Gaussian类别为q_i∈Δ⁴，原始语义通道为p_rc=Σ_i W_ri q_ic+t_r·1[c=bg]，t为固定残余背景。两臂共同采用明确的新诊断目标：固定δ=5e−7，令p̃_rc=(1−δ)p_rc+δ/5，最小化−Σ_r a_r log p̃_r,y_r。每view按其有效像素数平均，再对259张有标签TRAIN views等权；a_r包含同一固定class weight、valid/ignore规则和上述归一。没有final loss、Lovasz、教师或额外先验。δ事先固定，不根据数据调整；对渲染所得raw做仿射混合及log时用FP64聚合，不再clamp或归一化。

选择此方案而不新增一轮support筛查：q的ε内化不能保证零alpha且GT为前景的raw目标概率为正，而固定均匀噪声保证p̃≥δ/5=1e−7。此时零支持前景像素产生有限但与q无关的常数损失，不会凭空给Gaussian分配责任。它是与历史clamp/normalize CE不同的诊断目标，不能把结果直接解释为旧目标的优化收益。

若W真实且固定，标准E-step统计为

\[
M_{ic}=q_{ic}^{old}\sum_{r:y_r=c}\frac{a_r(1-\delta) W_{ri}}{\widetilde p_{r,y_r}}.
\]

残余背景(1−δ)t_r e_bg与新增均匀噪声δ/5都是不可更新的固定component，其责任不能划给Gaussian，包括Gaussian的背景类。整轮E-step必须保持同一个q_old，不能在view之间更新。可由**未归一raw通道经上述固定仿射混合**的加权负loglikelihood对颜色leaf的负梯度流式累计，再逐元素乘q_old得到M，不需要存储全图×全部Gaussian的W。必须保留(1−δ)因子，不能把噪声加入分母后仍使用旧公式。

不能对当前归一化p3d的loss直接取负梯度当计数：其ambient梯度额外含W_ri/(Σ_c raw_p_rc)，会产生非正计数。尽管理想simplex上的行和是常数，自动微分颜色leaf时不会自动施加该约束。历史−log(max(p,1e−7))也并非全域凸：floor处导数由0下降到−1/eps，不能套用凸性或标准EM推导。新的正仿射噪声目标在理想固定非负W下对q凸，但有限精度VJP/特征实现仍须实际核验；这不是历史clamp目标或神经参数目标的全局最优保证。必须报告raw/p̃有限性、零支持及噪声占目标概率的统计，不删去低支持样本。

当前gsplat反向已经发现tail transmittance重建误差。若VJP为k_r W_ri，累计量对应a_r k_r的另一个加权目标，而非原a_r目标。16-ray的行缩放校正不能外推为全259图校正。因此本方案只能把反向得到的统计视为**近似EM候选**；接受与否依赖原始渲染器完整TRAIN上固定仿射噪声目标的实际变化，不能从代理EM公式宣称单调。现有clamp/normalize后的p3d CE和评分只作描述，单独记录，避免与新目标混报。

## simplex下界与实际参数实现

统一预定ε=1e−5；两臂共同从q₀=(1−5ε)q_base+ε开始，映射后的实际FP32feature完全相同。记录理论q₀与GPU decoder实际FP32概率的误差；E-step的q_old必须使用后者，不能换成理论q₀或再夹紧后的值。保留原base→共同起点的参数/probability变化，以及E-pass测得的共同起点实际目标；固定预算内不额外测base完整TRAIN目标，不能声称已测初始化的目标变化，更不能把共同初始化算成优化收益。没有可见责任的全零M行保留q_old。

下界M-step解max Σ_c M_ic log q_ic，约束q_ic≥ε、Σ_cq_ic=1，满足q_ic=max(ε,M_ic/λ_i)，λ_i由行和等于1确定。必须用water-filling；任意clamp后再normalize不等价，也可能再次低于ε。

共同qε起点不等于共同可行域：标准feature Adam后续softmax分量可以低于ε，EM的下界M-step可行域更小。因此这是同起点/目标下两条优化路径的有限诊断，**不是同可行域的纯optimizer消融**。记录两臂实际最小q及低于ε的分量数；不为这一轮诊断另加受限Adam或改变既定范围。

decoder固定且其四行类差矩阵满秩。候选通过decoder伪逆映射为最小范数feature改变量；记录实际FP32softmax实现误差。FP64的M-step解和原渲染器浮点实现不是同一事物，不能免除真实目标检查。所有几何、SH、背景、decoder、refiner和camera张量前后逐位核验。

## 只保留一sweep/臂的最小比较

| 项目 | EM-style候选 | 匹配raw-only Adam |
|---|---|---|
| 起点 | 相同q₀和实际feature；相同固定decoder/refiner | 相同 |
| TRAIN数据/顺序 | 全259图一次、固定独立seed42顺序 | 相同 |
| 更新 | 完整E-pass后一次M-step | 每view一步，共259步 |
| 可变项 | 独立q，最后最小范数映回feature | 仅feature，新Adam；LR .01、eps1e−15、betas(.9,.999) |
| 目标 | 相同固定仿射噪声raw weighted per-view CE，δ=5e−7 | 相同 |
| 终点 | 固定这一轮候选，不搜索/重试/追加sweep | 固定第259步 |

EM的E-pass同时给出共同初始的实际渲染仿射噪声目标；两臂各另做一次完整TRAIN末态评价。合计**4×259=1,036次scene render**，不增加support预扫描，另记录实际backward、M-step/Adam更新数、rasterizer调用数和墙时；现有scene每次含RGB及semantic两个raster调用。两次末态评价可以在同次scene render读取raw/final输出，模型预测后按同一TRAIN评分规则统计。没有新增VAL、TTA或教师。

先比较真实全TRAIN目标。若候选不能可靠下降，保留原值及失败结果，不换步长、不多做M-step，不称其证明没有优化空间。若下降，也只说明这一轮重分配有可测优化收益。与Adam的比较是**同数据遍历/固定scene调用预算，不是同更新次数或同算力，也不是充分收敛比较**。

raw目标/评分是主诊断；final只记录冻结头受到的影响。最小范数映射和逐坐标Adam可能不同地改变16维feature及其decoder零空间，所以final变化不纯粹反映类别概率质量，更不能视为已适应新feature的最终部署结果。任何收益仍不证明全局最优、跨视图泛化或结构细化有益。

前提材料：[实际训练路径审查](raw_semantic_optimization_review.md)、[fixed-W先例和条件边界](semantic_feasibility_prior_art.md)、[16-ray诊断](semantic_compositing_lp_diagnostic.md)。
