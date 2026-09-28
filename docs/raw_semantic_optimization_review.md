# 原生语义低分的优化路径审查

2026-09-27，CPU只读核查，不修改模型、源码或训练设置，不运行GPU。对象为 `/mnt/data/SHM2026/runs/ssim_fixed_corner_v2_semantic_coupled/last.pt`，实际重新计算SHA256为 **391a0577450f7f458b75cb9e2fcf926953142b516728ea3600ef2ef064d72c13**。检查的是该run的冻结 `source_snapshot/bridge_rgs/{train,model,losses,semantic_schedule}.py`、checkpoint、warmstart RGB终点、既有8 TRAIN预测数组及日志；没有把当前main新增配置当成历史生效配置。

**没有发现可直接证实的训练缺陷；也没有证据证明固定可见性下的语义目标已经充分优化。**

## 已核实的训练合同

- 498,136个Gaussian；259张有标签TRAIN、原生分辨率、独立shuffle、固定8,000步。语义从第1步启用，geometry/RGB/camera冻结；teacher、fusion、semantic-to-geometry均关闭。
- warmstart保留全部非refiner张量，只重置refiner，随后创建新Adam，不加载RGB阶段optimizer。semantic prior counts与RGB起点逐位相同；该buffer不是本阶段持续施加的GT loss。初始概率来自平滑投票，未在warmstart误重置成随机值。
- checkpoint中feature与decoder的Adam step均为8,000；feature LR=.01、eps=1e−15，decoder LR=.001、eps=1e−8，refiner LR=.0003。没有启用semantic LR schedule。GT分支允许decoder梯度，`refiner_field_grad:true`允许最终输出通过原生概率基底和feature回传。
- 实际损失为final加权CE+0.2 Lovasz，再加0.5×raw加权CE；raw没有Lovasz项。两者class-weight power均为.25；按实际helper读取259个TRAIN mask，权重为 `[.456046,.922773,.853043,1.193589,1.574549]`。这是像素加权CE，不是类等权IoU；`.5`系数本身不能证明raw监督近乎关闭，Adam对整体梯度缩放并非线性等效。

上述路径见冻结 `train.py:479`（warmstart）、`:718`（GT损失）、`model.py:129`（分类/合成）、`:179`（optimizer）；主仓同名文件后续可能继续变化，历史解释以run快照为准。

## 末态与起点的CPU统计

以下概率分位数用保存的FP32参数在CPU FP64重算，属于Gaussian数量统计，**没有按像素贡献或可见性加权，也不代表3D类别真值**。

| 指标 | RGB warmstart起点 | 语义8k末态 |
|---|---:|---:|
| 每点最大类别概率中位数 | .764706 | .999116 |
| 每点类别熵中位数 | .871781 | .007594 |
| cable概率低于1e−4的点数 | 0 | 168,341（33.79%） |
| cable为最大概率类的点数 | 97,014 | 130,055（26.11%） |

末态19.82%的点最大概率≥.9999；CPU FP32计算无NaN/Inf、无概率精确为0，722个点最大概率因舍入为1。decoder四行类差矩阵满秩4，奇异值 `[7.061762,2.796158,2.245287,2.045200]`。因此没有“全部塌成背景”或decoder类差秩不足的证据；16维独立feature在这个固定decoder下可实现任意内部五类simplex概率。

41,183行feature（8.27%）与起点完全相同，其Adam二阶矩也全部为0。其余456,953行有变化；feature变化L2范数中位4.068。没有实际ray贡献证据将这41,183行关联到错误区域，不能把它们直接称为漏训的拉索点或证明不可见。

另读取已有8 TRAIN GPU预测数组，不新增渲染：764,951个GT拉索像素中493,035个raw预测错误；这些错误像素的cable概率中位数为.277437、最小.000464745。全部10,326,448个有效GT像素均未触及1e−7概率floor。因此这组像素没有支持“true-p clamp切断梯度”的解释。此统计与Gaussian自身softmax尖锐化是两个不同层次。[原诊断及数组来源](raw_semantic_mass_diagnostic_proposal.md)

## 尚未检验充分的瓶颈与解释限制

在固定W、floor不活跃时，单ray的raw CE对Gaussian logits满足（略去共同像素平均因子）

\[
\frac{\partial L}{\partial z_{ik}}
=c_y\frac{W_{ri}q_{i,y}}{p_{r,y}}\left(q_{i,k}-\mathbf 1[k=y]\right).
\]

这意味着纠正责任受当前目标类概率影响：即使某Gaussian有可见贡献，若它已确信错误类别，其目标类纠正责任也可能很小。末态尖锐化使这成为一个有数学依据的优化瓶颈假设；但没有误差ray的W、分项梯度或真实更新量，不能证明该机制正在主导。feature Adam的极小eps会重标度小梯度，也不能把小绝对梯度直接说成参数无法更新。

旧Rdetach只切断final对field的路径，仍使用同一softmax→概率合成及raw CE；它的raw cable约30.44%，coupled约29.79%，未显示明显解除瓶颈。旧常数/余弦LR控制只改后期步长，raw全类差+.1738pp且没有final收益，不证明不同参数化或固定W条件最优已达到。[历史配对](optimization_log.md)、[LR控制](semantic_convergence.md)

当前日志的`semantic_loss`混合final CE/Lovasz与0.5×raw CE，没有独立raw loss、贡献加权梯度、每点学习轨迹；不同随机视图的日志值也不是固定全集收敛曲线。已完成4-bias CE拟合下降而索IoU下降，进一步说明目标下降与该类IoU提高并不等价，但不证明全场CE最优。

少量固定TRAIN rays的条件可行assignment只能证明那些rays存在更好的共享赋值；它不说明完整259图上可兼容，也不能把标准convex fitting包装为创新。未发现明确bug时，不据此启动LR/温度/维度搜索或改训练；等待局部可见性诊断，并继续区分实现正确、条件表达能力和实际优化充分性。
