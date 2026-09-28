# Coupled 语义梯度诊断结果：不启动全投影训练

2026-09-27。**关闭本轮“将 final 的全部类别方向投掉以缓解有害干扰”的训练提案。** 这是当前证据不足的资源决策，不是所有梯度路由无效的结论。固定终点存在局部竞争及可测的null下降方向，但普通合梯度并未预测raw上升，feature路径整体贡献为正，未满足[实施前审查](/home/sky/workspace/SHM2026/docs/semantic_readout_gradient_routing_review.md)所需的稳定、有害feature干扰依据。本轮无训练、无性能提升，选型不变。

对象是实际coupled 8k终点 `runs/support_split_semantic_coupled/last.pt`（SHA `a329fc70d582972fc338715765304b99da208c821ee34288809db500d5acd6b9`），498136×16特征，无深度矩。完整259个有标签TRAIN各一次，002/205作固定重复及FD；仅求特征梯度，decoder／head／geometry／RGB冻结。原17模块snapshot和完整目标保持：`R=∇(.5 weighted raw CE)`；`F=∇(weighted final CE+.2 Lovasz+.001 residual²均值)`；`A`与F前向相同，只停止全部p3d通路梯度，保留rendered feature路径。下文横线表示等view均值，N是固定decoder类差矩阵的12维零空间投影。

| 统计 | 完整final F | feature-only A |
|---|---:|---:|
| `<R̄, ·̄>` | −3.107922e−11 | +1.746059e−11 |
| 与R̄的cosine | −0.0250603 | +0.0144603 |
| `mean_v <Rv, ·v>` | +8.319825e−9 | +4.420376e−10 |
| 逐视图负内积数，仅描述 | 23/259 | 138/259 |
| 全均值逐Gaussian正贡献质量 | 5.800907e−11 | 3.036433e−11 |
| 全均值逐Gaussian负贡献绝对质量 | 8.908829e−11 | 1.290374e−11 |

先平均梯度再求内积与先求逐视图内积再平均包含不同跨视图项，不能混用。A在138个视图上为负，不能称feature处处协同；但两种整体聚合净贡献均正，负视图数不能代替贡献大小。全均值旁路差额 `<R̄,F̄−Ā>=−4.853982e−11`，说明此条件导数分解的轻微负净内积来自p3d旁路差额；这不自动证明应训练旁路detach控制。

普通方向 `dc=−(R+F)` 在 **0/259** 视图上预测raw上升；全均值raw导数仍为 **−4.160988e−10**。合梯度范数 `6.184218e−5`，消减比 `||R̄+F̄||/(||R̄||+||F̄||)=0.775030`。负cos既不说明raw正在被推坏，也不说明两目标近乎完全抵消。

完整final全均值的null能量 `2.680142e−9`，约占77.92%，两个固定视图的真实final沿null下降通过FD。但候选 `dn=−(R+A N)` 只是另一种权衡：全均值raw导数约为 `−4.471780e−10`，相比普通方向稍强；真实final导数从 `−3.408357e−9`变为约`−2.649063e−9`，下降量少约22%。逐视图方向的平均则两目标下降量均减少。**未经等范数比较、未实际应用，不能称投影更优。** Adam预条件／moment与投影不交换，本轮没有验证其实际位移或有限更新的raw不变性。

数值与执行合同全部通过：固定两视图／三个h的 **18/18主FD** 可测，最大相对误差 **1.49934%**，最小信号／floor比67.488；6项完整final沿raw的交叉FD最大误差2.65818%。**6/6 raw沿null零控制**的实测中央差分均为0，实际FP32位移解析dot最大绝对值 `1.14773e−7`，均在预定绝对容差内；它们不是要求非零下降的主门。全均值raw-null、旁路差额-null相对L2分别 `3.30401e−7`、`1.02786e−7`，代数门通过。仅校准两固定视图的有限方向，不是全Jacobian证明。

实际285 scene／570 raster／783次特征VJP，44.749秒；0 optimizer step、0 checkpoint write、0 VAL／真实RGB像素解码。全部state／相机／flags／modes／原梯度恢复，源和输入未变。TF32及AMP按冻结合同关闭，原公式FP32、统计FP64，不声称逐位复现旧8k默认TF32执行。v1仅因CPU测试生成pytest cache而在来源核查停止（0 scene／VJP）；v2保存相同数值spec／源码／输入，仅更换执行目录，旧失败保留，未调门。

历史边界：源8k曾同时训练feature、decoder和head，而本轮冻结后两者，只识别终点条件导数，不能追溯原生类别退化的历史原因。随后H3是head-only并带深度矩；此前移除null特征的OOD消融也不是训练干扰证据。没有根据本结果另开旁路、阈值或步数搜索；负cos、非零null梯度与投影代数均不构成学术创新。

来源与独立核验：

- [v2 plan](/mnt/data/SHM2026/runs/coupled_semantic_gradient_diagnostic_v2/plan.json)：`14d2d34c393cd78602ba4944f79f7d19d879fafae3eaad491d5b727265b9e9ae`。
- [完成回执](/mnt/data/SHM2026/runs/coupled_semantic_gradient_diagnostic_v2/execution_receipt.json)：`04572e3948051c00d4a4dcca701b12c624b110d3a5bfd05494b8a22e9f832f7a`。
- [独立CPU统计复核](/mnt/data/SHM2026/runs/coupled_semantic_gradient_diagnostic_v2/independent_cpu_review.json)，已通过：`8c4e3d764441051ae68f2963f2355136087c13e984001f3a7152c8472861335f`。它核来源、标量恒等式、FD／零门及逐视图汇总；未保存梯度数组，不能从逐视图标量重建全均值向量，也未重做VJP。路由raw量用Cauchy小区间、final量用已存null范数极化得到；运行时恢复依据worker前后hash，不伪称独立重渲染。
