# 固定W的EM/FW结果：raw改善，冻结head下降，仍未收敛

2026-09-27。固定两block已自然完成，**20次EM与2次FW全部接受，26个完整pass后按原日程停止，不延长solver**。独立CPU审计passed。全部数据为259张拟合TRAIN的legacy网格，没有VAL、teacher或新模型采用；标准EM/FW和固定W属性求解不是学术创新。

本轮从[单方向诊断](simplex_direction_results.md)的唯一已确认FW候选开始，**不是从原H3重新初始化**。原H3 `22bc8a2d…5226`的几何、RGB、opacity、features、decoder、refiner和相机均冻结，仅外置498136×5的q更新。目标仍为原权重、逐view等权的affine-noise CE，δ=5e−7；head只替换p3d及log prior，保留原context。以下原H3读数由本次相同场/context实际记录，不能将本轮起点差异算作20EM+2FW的收益。

| 属性状态 | F | 近似全梯度FW gap |
|---|---:|---:|
| 原H3 | 0.103494906450 | 本轮未求原H3 gap |
| 本轮起点 | 0.103390119366 | 0.006798391223 |
| 本轮终点 | 0.102152275657 | 0.000644294851 |

F下降0.001237843709，相对1.197255%。终点重复F差0，但gap仍高于预定1e−5；`convergence_certified=false`，不能把剩余误差当作条件最优残差或几何容量上限。实际停止原因为 `fixed_two_block_schedule_finished`，不是收敛。

下表为pooled TRAIN CM的IoU，单位%；五类是等类均值，不是准确率。

| 状态与读出 | 五类mIoU | 背景 | 桥面 | 拉索 | 塔 | 基础 |
|---|---:|---:|---:|---:|---:|---:|
| 原H3 raw | 80.318479 | 93.916312 | 95.212682 | 34.630569 | 92.578845 | 85.253989 |
| 原H3 冻结head | 98.822236 | 99.848430 | 99.322428 | 98.669925 | 98.307287 | 97.963109 |
| 本轮起点 raw | 80.354421 | 93.929232 | 95.220024 | 34.729753 | 92.592423 | 85.300673 |
| 本轮起点 冻结head | 98.806726 | 99.847201 | 99.322138 | 98.662691 | 98.242113 | 97.959489 |
| 本轮终点 raw | 80.659340 | 94.046698 | 95.295547 | 35.367241 | 92.695238 | 85.891975 |
| 本轮终点 冻结head | 98.718592 | 99.841515 | 99.308481 | 98.640383 | 98.134793 | 97.667790 |

相对本轮起点，raw五类 **+0.304918pp**、拉索 **+0.637487pp**；冻结head五类 **-0.088134pp**。相对原H3，终点raw五类+0.340860pp、head-0.103643pp。属性目标的改善没有转为冻结读出改善；这支持检验新prior与head的兼容性，**不证明下降由失配单独造成**，也没有新视图结论。

成本来自[执行回执](/mnt/data/SHM2026/runs/raw_simplex_em_fw_v1/execution_receipt.json)及[自然完成记录](/mnt/data/SHM2026/runs/raw_simplex_em_fw_v1/launch_receipt.json)：内部1047.974846秒、外层1048.385646秒；6734 scene、13468 gsplat＋6734直接q shader，共20202 raster；6216 VJP、1036 head、518 mask/valid解码，0部分pass视图。allocated/reserved峰值1,847,072,768/2,204,106,752字节。耗时包含模型加载、全部遍历、缓存I/O/校验、FW搜索与评分，不是部署FPS。0模型参数更新/0生产checkpoint写入；独立q产物不支持普通resume。全部场状态、flags/梯度/数值设置恢复，输入源未变。

27项冻结CPU测试通过。[独立CPU审计](/mnt/data/SHM2026/runs/raw_simplex_em_fw_v1/independent_cpu_review.json)在0.489349秒内核69源、301输入，独立汇总保存的逐view F/CM、逐类IoU、22次接受记录、调用预算及q的simplex合同。它没有重新生成梯度或已释放target缓存，故未独立重算Wᵀ/gap/FW搜索；实际F与CM是保存记录复算，恢复与无额外活动是绑定回执核验，不是重放。审计SHA `9c2dcfc8daf108545f06c4c9fda5ccd93ddca007af667f7752b37f01fe67f743`。

来源：plan `29b066604846598a679a7fb0f1d25d88fce452cc25019adb3b552ad5a760fdcc`；[analysis](/mnt/data/SHM2026/runs/raw_simplex_em_fw_v1/analysis.json) SHA `8371430d1889e66215a48573222a7010ee24a8a98b28d8cbfd4f8c41115c13d1`；execution SHA `b129e790bb514dd1180e41978264af8b8ed6ee2ce9f297f24e2f17e749b9e7f0`。原PG未收敛与方向v1流程失败均保留，不追认通过。

下一步已决定准备一次固定两臂head适配：**原q＋原head续训2000步，对比本轮最终q＋同一原head续训2000步**；几何、features、q固定，样本/目标/优化预算一致，固定末态后统一官方评价。当前尚未完成适配或VAL评价，不预告收益；候选还须面对同预算控制及未续训原E，不能将避免继续训练退化当突破。E的模型与历史指标保持不变。
