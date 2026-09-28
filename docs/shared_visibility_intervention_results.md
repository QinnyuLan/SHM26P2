# 固定共享可见性干预：保存数组的CPU恢复结果

2026-09-27。**不采用固定近相机集合的opacity弱化，不据此启动长训或追加幅度／视角搜索。** 这轮只有有限TRAIN前向诊断，没有新模型或VAL成绩。原GPU尝试在最终JSON写入阶段退出1；264份预测数组已经保存，随后CPU恢复及另一独立实现的复算一致。下面只报告可从这些数组重建的测量，不把原运行改写成完整成功。

## 固定对象与结果

使用已完成的391a共享RGB／语义场，498,136点；集合G为到任一TRAIN相机距离小于`.01 × scene_scale`、且到最近初始化SfM点距离大于同一阈值的3,004点。集合只由几何定义，不是真实浮层标签。沿用既有8个TRAIN视角，原相机/native1320×989，关闭refiner与teacher。三种状态为原场、G的opacity logits减log(2)、加log(2)，各重复两次；分别从原值构造，不是直接把opacity乘.5或2。

完整场双通道前向给出原场G在每像素的合成贡献m，保留全部其他点的遮挡。主指标在相同`valid & known_label`支持上按固定m加权，每视角先聚合再等权平均。RGB用未夹紧float预测的MSE，语义用未归一raw概率的固定affine-noise CE；这些是TRAIN诊断量，不是官方PNG PSNR或最终mask指标。

| 损失，越低越好 | 原场 | 减log(2) | 加log(2) | 弱化相对原场 |
|---|---:|---:|---:|---:|
| 7可测视角贡献加权RGB MSE | .000321782858 | .000335271915 | .000444418913 | 恶化4.192% |
| 同一支持贡献加权raw CE | .129115871 | .130324490 | .128008297 | 恶化.936% |
| 全8视角完整valid RGB MSE | .000646478465 | .000652638619 | .000680196942 | 恶化.953% |
| 全8视角完整raw CE | .119772842 | .119276080 | .120782652 | 改善.415% |

7/8视角达到预定覆盖门；002视角G贡献为0，按原计划保留，没有替换样本。可测视角中只有167一张的两项加权loss同时下降，少于预定要求的6/7；加权RGB、加权语义相对改善门和整图RGB不退门也未过。因此判定为`specified_intervention_not_supported`，不因完整语义CE单独改善而采用。

描述性raw TRAIN mIoU为78.91845%→79.20532%→78.31949%，拉索IoU为28.42686%→28.39087%→28.44908%。弱化后的五类数值增益主要伴随基础类别改善，不能替代失败的连续主指标与RGB保护门，也不是最终refiner／DINOv3系统的性能。

该结论只针对固定G、固定±log(2)、固定8视角及当前颜色／类别赋值，不证明这些点物理上正确，也不否定其他几何调整。这里要求TRAIN不退是研究投入门，**不是VAL泛化改善的数学必要条件**；某种正则化仍可能牺牲训练拟合而改善泛化。本轮没有验证这种可能性，也没有建立学术贡献。

## 故障、修复与证据边界

原GPU进程在最后`json.dumps`保存报告时失败，父回执退出1、耗时9.2528秒。原因是重复误差完全相同时，容差下限保留为NumPy浮点，两个比较结果因此成为NumPy布尔值，标准JSON编码器拒绝处理。它不改变测量公式或门槛，但阻止了完整运行报告保存。原冻结快照、plan、失败回执与日志保持原样，未重新运行GPU。

工作区将容差显式转Python float、两个判据显式转bool，并新增真实score→coverage→summary→restore→严格JSON回归；25项worker合同和5项恢复合同通过。CPU恢复另建不可变计划，绑定原失败链及264数组SHA；复用原冻结数值函数，JSON只将NumPy标量转为对应Python标量，继续拒绝NaN。CPU恢复耗时7.1973秒，新增render/backward/optimizer均为0。

独立CPU审核没有调用恢复入口的评分／gate函数：重新推导G3004、核验全部264数组及来源，并重算贡献支持、三种状态loss、混淆矩阵、重复容差和全部判据，与恢复结果一致。直接全点颜色质量与alpha最大差1.19209e−6，小于原定5e−6；RGB、语义与贡献渲染alpha最大差0，保存的raw IDs与重新argmax完全一致。

原进程没有成功持久化的全state恢复哈希、实际raster执行计数与预测／GT阶段时间戳均明确为缺失。**104次raster是冻结代码与完整保存集合所对应的计划数量，不能当作成功保存的运行时计数证明。** 当前输入checkpoint、源码和数组SHA可以核验，但不伪造原进程的恢复回执；这次有限数值否定结论不等于原GPU合同全部通过。

## 可复核材料

- [固定诊断设计](next_shared_geometry_hypothesis.md)；[原计划](/mnt/data/SHM2026/runs/shared_visibility_intervention_v1/plan.json)，SHA `8aa0538bb24c9f6189462d79fdf2a77cfc9fe734fcd76d00857b82090badb7fd`。
- [原失败回执](/mnt/data/SHM2026/runs/shared_visibility_intervention_v1/execution_receipt.json)、[原日志](/mnt/data/SHM2026/runs/shared_visibility_intervention_v1/worker.log)。
- [CPU恢复计划](/mnt/data/SHM2026/runs/shared_visibility_intervention_v1/recovery/plan.json)，SHA `3762e6ef6dec8eaea5f81a5d4130db3ab71fdadc3c2cc3669d866f8903406589`。
- [恢复测量](/mnt/data/SHM2026/runs/shared_visibility_intervention_v1/recovery/recovered_measurements.json)，SHA `64340e7915bcde859eea81798e0b7c14aa8703bffde79aac12f558948cdec67a`。
- [独立CPU核验](/mnt/data/SHM2026/runs/shared_visibility_intervention_v1/recovery/independent_cpu_result_audit.json)，SHA `65baaf666430e409bd8347211260e23214ac7c402e0fbb15c823891cfcfe55a0`。
